import asyncio
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.base import get_provider
from agents.discrepancy_agent import run_comparison
from api.dependencies import require_roles, CurrentUser
from db.models import Invoice, Extraction
from db.session import get_db
from models.invoice import InvoiceSchema

router = APIRouter(tags=["compare"])
log = structlog.get_logger()


class CompareRequest(BaseModel):
    invoice_ids: list[uuid.UUID]


@router.post("/compare", response_model=dict)
async def compare_invoices(
    body: CompareRequest,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    if len(body.invoice_ids) < 2:
        raise HTTPException(400, "Need at least 2 invoice IDs to compare")

    named_schemas: list[tuple[str, InvoiceSchema]] = []
    for inv_id in body.invoice_ids:
        inv = await db.scalar(
            select(Invoice).where(Invoice.id == inv_id, Invoice.tenant_id == uuid.UUID(user.tenant_id))
        )
        if not inv:
            raise HTTPException(404, f"Invoice {inv_id} not found")
        ext = await db.scalar(select(Extraction).where(Extraction.invoice_id == inv_id))
        if not ext:
            raise HTTPException(409, f"Invoice {inv_id} has no extraction — run /extract first")
        named_schemas.append((inv.file_name, InvoiceSchema.model_validate(ext.schema_json)))

    provider = get_provider()
    result = await asyncio.to_thread(run_comparison, named_schemas, provider)
    return {"data": result, "error": None, "request_id": None}
