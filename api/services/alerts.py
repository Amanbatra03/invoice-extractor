import asyncio
import hashlib

import structlog
from redis import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import get_settings
from api.dependencies import get_queue
from db.models import Alert

log = structlog.get_logger()

_DETAIL_MAX = 2000


def _fingerprint(source: str, event: str) -> str:
    return hashlib.sha256(f"{source}:{event}".encode()).hexdigest()[:16]


async def raise_alert(
    db: AsyncSession,
    *,
    severity: str,
    source: str,
    event: str,
    detail: str,
    context: dict | None = None,
) -> Alert | None:
    """Persist an alert and (unless suppressed) enqueue Discord dispatch.

    Never raises: an alerting failure must not take down the failing
    component that is being reported.
    """
    try:
        settings = get_settings()
        fp = _fingerprint(source, event)
        alert = Alert(
            severity=severity,
            source=source,
            event=event,
            detail=(detail or "")[:_DETAIL_MAX],
            context=context,
            fingerprint=fp,
        )

        if not settings.ALERT_DISCORD_WEBHOOK_URL:
            alert.delivery_status = "skipped"
            db.add(alert)
            await db.commit()
            return alert

        suppressed = False
        try:
            conn = Redis.from_url(settings.REDIS_URL)
            acquired = conn.set(
                f"alert:cd:{fp}", "1", nx=True, ex=settings.ALERT_COOLDOWN_SECONDS
            )
            suppressed = not acquired
        except Exception as exc:
            # Fail open: no cooldown info means we'd rather alert than stay silent.
            log.warning("alert.cooldown_check_failed", error=str(exc))

        alert.delivery_status = "suppressed" if suppressed else "pending"
        db.add(alert)
        await db.commit()

        if not suppressed:
            try:
                get_queue().enqueue("workers.alert_job.run", str(alert.id))
            except Exception as exc:
                log.error("alert.enqueue_failed", alert_id=str(alert.id), error=str(exc))
        return alert

    except Exception as exc:
        log.error("alert.raise_failed", error=str(exc), alert_source=source, alert_event=event)
        return None


def raise_alert_sync(
    *,
    severity: str,
    source: str,
    event: str,
    detail: str,
    context: dict | None = None,
) -> None:
    """Wrapper for synchronous contexts (RQ exception handler)."""

    async def _run() -> None:
        from db.session import get_session_factory
        async with get_session_factory()() as db:
            await raise_alert(
                db, severity=severity, source=source, event=event,
                detail=detail, context=context,
            )

    try:
        asyncio.run(_run())
    except Exception as exc:
        log.error("alert.raise_sync_failed", error=str(exc), source=source, event=event)
