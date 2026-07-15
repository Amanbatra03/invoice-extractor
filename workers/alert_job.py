import asyncio
import uuid
from datetime import datetime, timezone

import httpx
import structlog
from sqlalchemy import select

from api.config import get_settings
from db.models import Alert
from db.session import get_session_factory

log = structlog.get_logger()

_RETRY_DELAYS = [5, 30, 120]
_MAX_ATTEMPTS = len(_RETRY_DELAYS) + 1
_SEVERITY_COLORS = {"error": 0xE74C3C, "warning": 0xF39C12}


def _build_embed(alert: Alert, env: str) -> dict:
    fields = [
        {"name": "Source", "value": alert.source, "inline": True},
        {"name": "Severity", "value": alert.severity, "inline": True},
        {"name": "Detail", "value": (alert.detail or "-")[:1000], "inline": False},
    ]
    context_lines = "\n".join(f"**{k}**: {v}" for k, v in (alert.context or {}).items())
    if context_lines:
        fields.append({"name": "Context", "value": context_lines[:1000], "inline": False})
    created = alert.created_at or datetime.now(timezone.utc)
    return {
        "embeds": [{
            "title": f"[{env}] {alert.severity.upper()} — {alert.event}"[:256],
            "color": _SEVERITY_COLORS.get(alert.severity, 0x95A5A6),
            "fields": fields,
            "timestamp": created.isoformat(),
        }]
    }


def run(alert_id_str: str) -> None:
    asyncio.run(_run_async(alert_id_str))


async def _run_async(alert_id_str: str) -> None:
    settings = get_settings()
    alert_id = uuid.UUID(alert_id_str)
    session_factory = get_session_factory()

    async with session_factory() as db:
        alert = await db.scalar(select(Alert).where(Alert.id == alert_id))
        if not alert:
            log.warning("alert.not_found", alert_id=alert_id_str)
            return
        if not settings.ALERT_DISCORD_WEBHOOK_URL:
            alert.delivery_status = "skipped"
            await db.commit()
            return

        payload = _build_embed(alert, settings.ENV)

        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if attempt > 0:
                await asyncio.sleep(delay)
            alert.delivery_attempts = attempt + 1
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.post(
                        settings.ALERT_DISCORD_WEBHOOK_URL, json=payload
                    )
                if resp.is_success:
                    alert.delivery_status = "delivered"
                    alert.delivered_at = datetime.now(timezone.utc)
                    await db.commit()
                    log.info("alert.delivered", alert_id=alert_id_str, attempt=attempt + 1)
                    return
                alert.last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                await db.commit()
                log.warning("alert.delivery_retry", attempt=attempt + 1, status=resp.status_code)
            except Exception as exc:
                alert.last_error = str(exc)[:500]
                await db.commit()
                log.warning("alert.delivery_error", attempt=attempt + 1, error=str(exc))

        alert.delivery_status = "failed"
        await db.commit()
        log.error("alert.delivery_failed", alert_id=alert_id_str)
