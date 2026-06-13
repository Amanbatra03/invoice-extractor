import asyncio
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser, get_queue
from api.schemas.extraction import ExtractionOut, ValidationResult
from agents.validation_agent import run_validation
from db.models import Invoice, Extraction, Job
from db.session import get_db
from models.invoice import InvoiceSchema

router = APIRouter(tags=["extraction"])
log = structlog.get_logger()


async def _get_invoice(invoice_id: uuid.UUID, tenant_id: str, db: AsyncSession) -> Invoice:
    inv = await db.scalar(
        select(Invoice).where(Invoice.id == invoice_id, Invoice.tenant_id == uuid.UUID(tenant_id))
    )
    if not inv:
        raise HTTPException(404, "Invoice not found")
    return inv


async def _enqueue_extract(invoice_id: uuid.UUID, tenant_id: str, db: AsyncSession, queue) -> uuid.UUID:
    job = Job(
        tenant_id=uuid.UUID(tenant_id),
        type="extract",
        status="queued",
        payload={"invoice_id": str(invoice_id)},
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    queue.enqueue("workers.extract_job.run", str(invoice_id), str(job.id), job_timeout=120)
    return job.id


@router.post("/invoices/{invoice_id}/extract", response_model=dict)
async def run_extraction(
    invoice_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "api_user")),
    db: AsyncSession = Depends(get_db),
    queue=Depends(get_queue),
):
    inv = await _get_invoice(invoice_id, user.tenant_id, db)
    if inv.status not in ("ready", "failed"):
        raise HTTPException(409, f"Invoice status is '{inv.status}', must be 'ready' to extract")
    job_id = await _enqueue_extract(invoice_id, user.tenant_id, db, queue)
    log.info("extraction.queued", invoice_id=str(invoice_id), job_id=str(job_id))
    return {"data": {"job_id": job_id, "status": "queued"}, "error": None, "request_id": None}


@router.get("/invoices/{invoice_id}/extraction", response_model=dict)
async def get_extraction(
    invoice_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    await _get_invoice(invoice_id, user.tenant_id, db)
    ext = await db.scalar(
        select(Extraction).where(
            Extraction.invoice_id == invoice_id,
            Extraction.tenant_id == uuid.UUID(user.tenant_id),
        )
    )
    if not ext:
        raise HTTPException(404, "No extraction found — run POST /extract first")
    return {"data": ExtractionOut.model_validate(ext), "error": None, "request_id": None}


@router.post("/invoices/{invoice_id}/validate", response_model=dict)
async def validate_extraction(
    invoice_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst")),
    db: AsyncSession = Depends(get_db),
):
    await _get_invoice(invoice_id, user.tenant_id, db)
    ext = await db.scalar(
        select(Extraction).where(
            Extraction.invoice_id == invoice_id,
            Extraction.tenant_id == uuid.UUID(user.tenant_id),
        )
    )
    if not ext:
        raise HTTPException(404, "No extraction found")
    schema = InvoiceSchema.model_validate(ext.schema_json)
    report = await asyncio.to_thread(run_validation, schema)
    return {
        "data": ValidationResult(passed=report.passed, issues=report.issues),
        "error": None,
        "request_id": None,
    }
