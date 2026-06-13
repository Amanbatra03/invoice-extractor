import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser
from api.schemas.webhook import WebhookIn, WebhookOut, WebhookPatch
from api.services.webhook_signer import build_webhook_payload, sign_payload
from db.models import Webhook, WebhookDelivery
from db.session import get_db

router = APIRouter(prefix="/webhooks", tags=["webhooks"])
log = structlog.get_logger()


@router.post("", response_model=dict)
async def create_webhook(
    body: WebhookIn,
    user: CurrentUser = Depends(require_roles("admin", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    wh = Webhook(
        tenant_id=uuid.UUID(user.tenant_id),
        url=body.url,
        events=body.events,
        secret=body.secret,
        active=True,
        created_at=datetime.now(timezone.utc),
    )
    db.add(wh)
    await db.commit()
    await db.refresh(wh)
    return {"data": WebhookOut.model_validate(wh), "error": None, "request_id": None}


@router.get("", response_model=dict)
async def list_webhooks(
    user: CurrentUser = Depends(require_roles("admin", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(Webhook).where(Webhook.tenant_id == uuid.UUID(user.tenant_id))
    )).scalars().all()
    return {"data": [WebhookOut.model_validate(r) for r in rows], "error": None, "request_id": None}


@router.get("/{webhook_id}", response_model=dict)
async def get_webhook(
    webhook_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    wh = await db.scalar(
        select(Webhook).where(Webhook.id == webhook_id, Webhook.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not wh:
        raise HTTPException(404, "Webhook not found")
    deliveries = (await db.execute(
        select(WebhookDelivery)
        .where(WebhookDelivery.webhook_id == webhook_id)
        .order_by(WebhookDelivery.delivered_at.desc())
        .limit(20)
    )).scalars().all()
    return {
        "data": {
            **WebhookOut.model_validate(wh).model_dump(),
            "deliveries": [
                {
                    "event": d.event,
                    "status": d.status,
                    "attempts": d.attempts,
                    "last_error": d.last_error,
                }
                for d in deliveries
            ],
        },
        "error": None,
        "request_id": None,
    }


@router.patch("/{webhook_id}", response_model=dict)
async def update_webhook(
    webhook_id: uuid.UUID,
    body: WebhookPatch,
    user: CurrentUser = Depends(require_roles("admin", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    wh = await db.scalar(
        select(Webhook).where(Webhook.id == webhook_id, Webhook.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not wh:
        raise HTTPException(404, "Webhook not found")
    if body.url is not None:
        wh.url = body.url
    if body.events is not None:
        wh.events = body.events
    if body.active is not None:
        wh.active = body.active
    await db.commit()
    await db.refresh(wh)
    return {"data": WebhookOut.model_validate(wh), "error": None, "request_id": None}


@router.delete("/{webhook_id}", response_model=dict)
async def delete_webhook(
    webhook_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    wh = await db.scalar(
        select(Webhook).where(Webhook.id == webhook_id, Webhook.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not wh:
        raise HTTPException(404, "Webhook not found")
    await db.delete(wh)
    await db.commit()
    return {"data": {"deleted": True}, "error": None, "request_id": None}


@router.post("/{webhook_id}/test", response_model=dict)
async def test_webhook(
    webhook_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    import httpx

    wh = await db.scalar(
        select(Webhook).where(Webhook.id == webhook_id, Webhook.tenant_id == uuid.UUID(user.tenant_id))
    )
    if not wh:
        raise HTTPException(404, "Webhook not found")
    payload = build_webhook_payload(
        "webhook.test", user.tenant_id, {"message": "Test ping from Invoice Analyst"}
    )
    signature = sign_payload(payload, wh.secret)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(wh.url, json=payload, headers={"X-Signature": signature})
        return {"data": {"status": resp.status_code, "ok": resp.is_success}, "error": None, "request_id": None}
    except Exception as exc:
        return {"data": {"status": 0, "ok": False}, "error": str(exc), "request_id": None}
