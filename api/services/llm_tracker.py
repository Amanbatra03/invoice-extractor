import time
import uuid
import functools
from datetime import datetime, timezone
from typing import Callable

import structlog

log = structlog.get_logger()

_COST_PER_INPUT_TOKEN = 0.000_000_10
_COST_PER_OUTPUT_TOKEN = 0.000_000_40


def llm_usage_tracker(agent: str):
    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(*args, tenant_id: str, invoice_id: str | None = None, db=None, **kwargs):
            start = time.monotonic()
            result = await fn(*args, tenant_id=tenant_id, invoice_id=invoice_id, db=db, **kwargs)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            if db is not None:
                try:
                    from db.models import LlmUsage
                    usage = LlmUsage(
                        tenant_id=uuid.UUID(tenant_id),
                        invoice_id=uuid.UUID(invoice_id) if invoice_id else None,
                        model="gemini-2.0-flash",
                        agent=agent,
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=elapsed_ms,
                        cost_usd=0,
                        created_at=datetime.now(timezone.utc),
                    )
                    db.add(usage)
                    await db.commit()
                except Exception as exc:
                    log.warning("llm_usage_write_failed", error=str(exc))
            return result
        return wrapper
    return decorator
