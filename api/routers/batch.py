import asyncio
import io
import uuid
from datetime import datetime, timezone

import pandas as pd
import structlog
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import require_roles, CurrentUser, get_queue
from db.models import Invoice, Job, Extraction
from db.session import get_db
from models.invoice import InvoiceSchema

router = APIRouter(prefix="/batch", tags=["batch"])
log = structlog.get_logger()


class BatchExtractRequest(BaseModel):
    invoice_ids: list[uuid.UUID]


async def _all_invoices_exist(
    invoice_ids: list[uuid.UUID], tenant_id: str, db: AsyncSession
) -> bool:
    for inv_id in invoice_ids:
        inv = await db.scalar(
            select(Invoice).where(
                Invoice.id == inv_id,
                Invoice.tenant_id == uuid.UUID(tenant_id),
            )
        )
        if not inv:
            raise HTTPException(404, f"Invoice {inv_id} not found")
    return True


@router.post("/extract", response_model=dict)
async def batch_extract(
    body: BatchExtractRequest,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "api_user")),
    db: AsyncSession = Depends(get_db),
    queue=Depends(get_queue),
):
    if not body.invoice_ids:
        raise HTTPException(400, "invoice_ids must not be empty")
    await _all_invoices_exist(body.invoice_ids, user.tenant_id, db)

    job = Job(
        tenant_id=uuid.UUID(user.tenant_id),
        type="batch_extract",
        status="queued",
        payload={"invoice_ids": [str(i) for i in body.invoice_ids]},
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    queue.enqueue("workers.batch_job.run", str(job.id), job_timeout=1800)
    log.info("batch.queued", job_id=str(job.id), count=len(body.invoice_ids))
    return {
        "data": {"batch_job_id": job.id, "status": "queued"},
        "error": None,
        "request_id": None,
    }


@router.get("/{job_id}", response_model=dict)
async def get_batch_status(
    job_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    job = await db.scalar(
        select(Job).where(
            Job.id == job_id,
            Job.tenant_id == uuid.UUID(user.tenant_id),
            Job.type == "batch_extract",
        )
    )
    if not job:
        raise HTTPException(404, "Batch job not found")
    return {
        "data": {
            "batch_job_id": job.id,
            "status": job.status,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        },
        "error": None,
        "request_id": None,
    }


@router.get("/{job_id}/export")
async def export_batch_csv(
    job_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    job = await db.scalar(
        select(Job).where(
            Job.id == job_id,
            Job.tenant_id == uuid.UUID(user.tenant_id),
        )
    )
    if not job or not job.result:
        raise HTTPException(404, "Batch job not found or not yet complete")

    invoice_ids = job.payload.get("invoice_ids", [])
    rows = []
    for inv_id in invoice_ids:
        ext = await db.scalar(
            select(Extraction).where(
                Extraction.invoice_id == uuid.UUID(inv_id),
                Extraction.tenant_id == uuid.UUID(user.tenant_id),
            )
        )
        inv = await db.scalar(
            select(Invoice).where(
                Invoice.id == uuid.UUID(inv_id),
                Invoice.tenant_id == uuid.UUID(user.tenant_id),
            )
        )
        if ext and inv:
            schema = InvoiceSchema.model_validate(ext.schema_json)
            row = {"invoice": inv.file_name, **schema.model_dump(exclude={"line_items"})}
            rows.append(row)

    df = await asyncio.to_thread(pd.DataFrame, rows)
    buf = io.StringIO()
    await asyncio.to_thread(lambda: df.to_csv(buf, index=False))
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=batch_{job_id}.csv"},
    )
