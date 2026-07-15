import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import select, update

from agents.base import get_provider
from agents.batch_agent import run_batch
from db.models import Job
from db.session import get_session_factory

log = structlog.get_logger()


def run(job_id_str: str) -> None:
    import asyncio
    asyncio.run(_run_async(job_id_str))


async def _run_async(job_id_str: str) -> None:
    job_id = uuid.UUID(job_id_str)
    session_factory = get_session_factory()

    async with session_factory() as db:
        job = await db.scalar(select(Job).where(Job.id == job_id))
        if not job:
            log.error("batch.job_not_found", job_id=job_id_str)
            return

        invoice_ids: list[str] = job.payload.get("invoice_ids", [])
        await db.execute(update(Job).where(Job.id == job_id).values(status="running"))
        await db.commit()

        try:
            provider = get_provider()
            results = await run_batch(invoice_ids, db, provider)
            await db.execute(
                update(Job).where(Job.id == job_id).values(
                    status="done",
                    result={
                        "success_count": results["success_count"],
                        "failure_count": results["failure_count"],
                        "failed_ids": list(results["failed"].keys()),
                        "errors": results["failed"],
                    },
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()
            log.info("batch.done", job_id=job_id_str, done=results["success_count"], failed=results["failure_count"])

        except Exception as exc:
            log.error("batch.failed", job_id=job_id_str, error=str(exc))
            await db.execute(
                update(Job).where(Job.id == job_id).values(
                    status="failed", error=str(exc), completed_at=datetime.now(timezone.utc)
                )
            )
            await db.commit()
            raise
