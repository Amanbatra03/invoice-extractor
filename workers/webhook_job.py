import asyncio
import uuid
from datetime import datetime, timezone

import httpx
import structlog
from sqlalchemy import select

from api.services.webhook_signer import sign_payload
from db.models import Webhook, WebhookDelivery
from db.session import get_session_factory

log = structlog.get_logger()

_RETRY_DELAYS = [5, 30, 120, 600, 1800]
_MAX_ATTEMPTS = len(_RETRY_DELAYS) + 1


def _build_signed_request(payload: dict, secret: str) -> dict:
    signature = sign_payload(payload, secret)
    return {
        "Content-Type": "application/json",
        "X-Signature": signature,
        "X-Webhook-Event": payload.get("event", ""),
    }


def run(webhook_id_str: str, event: str, payload: dict) -> None:
    asyncio.run(_run_async(webhook_id_str, event, payload))


async def _run_async(webhook_id_str: str, event: str, payload: dict) -> None:
    webhook_id = uuid.UUID(webhook_id_str)
    session_factory = get_session_factory()

    async with session_factory() as db:
        wh = await db.scalar(select(Webhook).where(Webhook.id == webhook_id, Webhook.active == True))
        if not wh:
            log.warning("webhook.not_found_or_inactive", webhook_id=webhook_id_str)
            return

        delivery = WebhookDelivery(
            webhook_id=webhook_id,
            event=event,
            payload=payload,
            status="pending",
            attempts=0,
        )
        db.add(delivery)
        await db.commit()
        await db.refresh(delivery)

        headers = _build_signed_request(payload, wh.secret)

        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if attempt > 0:
                await asyncio.sleep(delay)

            delivery.attempts = attempt + 1
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(wh.url, json=payload, headers=headers)
                if resp.is_success:
                    delivery.status = "delivered"
                    delivery.delivered_at = datetime.now(timezone.utc)
                    await db.commit()
                    log.info("webhook.delivered", webhook_id=webhook_id_str, event=event, attempt=attempt + 1)
                    return
                else:
                    delivery.last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    await db.commit()
                    log.warning("webhook.delivery_failed", attempt=attempt + 1, status=resp.status_code)

            except Exception as exc:
                delivery.last_error = str(exc)[:500]
                await db.commit()
                log.warning("webhook.delivery_error", attempt=attempt + 1, error=str(exc))

            if attempt + 1 >= _MAX_ATTEMPTS:
                delivery.status = "failed"
                await db.commit()
                log.error("webhook.permanently_failed", webhook_id=webhook_id_str, event=event)
