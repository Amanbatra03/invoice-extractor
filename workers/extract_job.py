import asyncio
import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import select, update

from agents.base import get_provider
from agents.extraction_agent import run_extraction
from agents.retriever import HybridRetriever
from agents.validation_agent import run_validation
from db.models import Invoice, Extraction, Job
from db.session import get_session_factory

log = structlog.get_logger()


def run(invoice_id_str: str, job_id_str: str) -> None:
    import asyncio
    asyncio.run(_run_async(invoice_id_str, job_id_str))


async def _run_async(invoice_id_str: str, job_id_str: str) -> None:
    invoice_id = uuid.UUID(invoice_id_str)
    job_id = uuid.UUID(job_id_str)
    session_factory = get_session_factory()

    async with session_factory() as db:
        inv = await db.scalar(select(Invoice).where(Invoice.id == invoice_id))
        if not inv:
            log.error("extract.invoice_not_found", invoice_id=invoice_id_str)
            return

        await db.execute(update(Job).where(Job.id == job_id).values(status="running"))
        await db.commit()

        try:
            provider = get_provider()

            if inv.file_type == "image":
                from api.services.storage import download_file
                import tempfile
                from pathlib import Path
                content = await asyncio.to_thread(download_file, str(inv.tenant_id), inv.storage_path)
                suffix = "." + inv.file_name.rsplit(".", 1)[-1]
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = Path(tmp.name)
                raw = provider.generate_structured(
                    "Extract all invoice fields from this image.", type("ImageSchema", (), {})
                )
                from models.invoice import InvoiceSchema
                schema = InvoiceSchema.model_validate(raw)
                tmp_path.unlink(missing_ok=True)
            else:
                retriever = HybridRetriever(invoice_id=invoice_id, db=db, provider=provider)
                schema = await run_extraction(retriever, provider)

            validation_report = run_validation(schema)

            existing = await db.scalar(select(Extraction).where(Extraction.invoice_id == invoice_id))
            if existing:
                existing.schema_json = schema.model_dump()
                existing.model_used = "gemini-2.0-flash"
                existing.validated = validation_report.passed
            else:
                db.add(Extraction(
                    invoice_id=invoice_id,
                    tenant_id=inv.tenant_id,
                    schema_json=schema.model_dump(),
                    model_used="gemini-2.0-flash",
                    validated=validation_report.passed,
                    created_at=datetime.now(timezone.utc),
                ))

            await db.execute(
                update(Job).where(Job.id == job_id).values(
                    status="done",
                    result={"validated": validation_report.passed, "issues": validation_report.issues},
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()
            log.info("extract.done", invoice_id=invoice_id_str)

        except Exception as exc:
            log.error("extract.failed", invoice_id=invoice_id_str, error=str(exc))
            await db.execute(
                update(Job).where(Job.id == job_id).values(
                    status="failed", error=str(exc), completed_at=datetime.now(timezone.utc)
                )
            )
            await db.commit()
            raise
