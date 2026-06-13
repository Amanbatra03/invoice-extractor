import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import get_current_user, require_roles, CurrentUser, get_queue
from api.schemas.invoice import InvoiceOut, InvoiceUploadResponse, InvoiceListResponse
from api.services.storage import upload_file, get_signed_url, delete_file, sha256_file
from db.models import Invoice, Job
from db.session import get_db

router = APIRouter(prefix="/invoices", tags=["invoices"])
log = structlog.get_logger()

_ALLOWED_TYPES = {"application/pdf", "image/jpeg", "image/png", "image/jpg"}
_MAX_SIZE = 50 * 1024 * 1024  # 50 MB


def _detect_file_type(filename: str, content: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return "pdf"
    if ext in ("jpg", "jpeg", "png"):
        return "image"
    raise HTTPException(400, f"Unsupported file type: .{ext}")


def _validate_upload(file: UploadFile, content: bytes) -> str:
    if len(content) > _MAX_SIZE:
        raise HTTPException(400, "File exceeds 50MB limit")
    if file.content_type and file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported content type: {file.content_type}")
    if content[:4] == b"MZ\x90\x00" or content[:2] == b"MZ":
        raise HTTPException(400, "Executable files are not allowed")
    return _detect_file_type(file.filename or "", content)


async def _enqueue_ingest(invoice_id: uuid.UUID, job_id: uuid.UUID, queue) -> uuid.UUID:
    queue.enqueue(
        "workers.ingest_job.run",
        str(invoice_id),
        str(job_id),
        job_timeout=300,
    )
    return job_id


@router.post("/upload", response_model=dict)
async def upload_invoice(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_roles("admin", "analyst", "api_user")),
    db: AsyncSession = Depends(get_db),
    queue=Depends(get_queue),
):
    content = await file.read()
    file_type = _validate_upload(file, content)
    sha = sha256_file(content)

    existing = await db.scalar(
        select(Invoice).where(
            Invoice.tenant_id == uuid.UUID(user.tenant_id),
            Invoice.sha256 == sha,
        )
    )
    if existing:
        return {
            "data": {"invoice_id": existing.id, "job_id": None, "status": "already_exists"},
            "error": None,
            "request_id": None,
        }

    storage_path = upload_file(user.tenant_id, file.filename or f"upload.{file_type}", content)
    invoice = Invoice(
        tenant_id=uuid.UUID(user.tenant_id),
        uploaded_by=uuid.UUID(user.id) if user.id and user.id != "api_key" else None,
        file_name=file.filename or f"upload.{file_type}",
        file_type=file_type,
        storage_path=storage_path,
        sha256=sha,
        status="pending",
        created_at=datetime.now(timezone.utc),
    )
    db.add(invoice)
    await db.flush()

    job = Job(
        tenant_id=uuid.UUID(user.tenant_id),
        type="ingest",
        status="queued",
        payload={"invoice_id": str(invoice.id)},
        created_at=datetime.now(timezone.utc),
    )
    db.add(job)
    await db.commit()
    await db.refresh(invoice)

    await _enqueue_ingest(invoice.id, job.id, queue)
    log.info("invoice.uploaded", invoice_id=str(invoice.id), tenant_id=user.tenant_id)
    return {
        "data": InvoiceUploadResponse(invoice_id=invoice.id, job_id=job.id, status="ingesting"),
        "error": None,
        "request_id": None,
    }


@router.get("", response_model=dict)
async def list_invoices(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    status: str | None = Query(None),
    file_type: str | None = Query(None),
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    q = select(Invoice).where(Invoice.tenant_id == uuid.UUID(user.tenant_id))
    if status:
        q = q.where(Invoice.status == status)
    if file_type:
        q = q.where(Invoice.file_type == file_type)
    total = await db.scalar(select(func.count()).select_from(q.subquery()))
    rows = (await db.execute(q.offset((page - 1) * limit).limit(limit))).scalars().all()
    return {
        "data": InvoiceListResponse(
            items=[InvoiceOut.model_validate(r) for r in rows],
            total=total or 0,
            page=page,
            limit=limit,
        ),
        "error": None,
        "request_id": None,
    }


@router.get("/{invoice_id}", response_model=dict)
async def get_invoice(
    invoice_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer", "api_user")),
    db: AsyncSession = Depends(get_db),
):
    inv = await db.scalar(
        select(Invoice).where(
            Invoice.id == invoice_id,
            Invoice.tenant_id == uuid.UUID(user.tenant_id),
        )
    )
    if not inv:
        raise HTTPException(404, "Invoice not found")
    return {"data": InvoiceOut.model_validate(inv), "error": None, "request_id": None}


@router.get("/{invoice_id}/download", response_model=dict)
async def download_invoice(
    invoice_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer")),
    db: AsyncSession = Depends(get_db),
):
    inv = await db.scalar(
        select(Invoice).where(
            Invoice.id == invoice_id,
            Invoice.tenant_id == uuid.UUID(user.tenant_id),
        )
    )
    if not inv:
        raise HTTPException(404, "Invoice not found")
    url = get_signed_url(user.tenant_id, inv.storage_path)
    return {"data": {"signed_url": url, "expires_in": 900}, "error": None, "request_id": None}


@router.delete("/{invoice_id}", response_model=dict)
async def delete_invoice(
    invoice_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin")),
    db: AsyncSession = Depends(get_db),
):
    inv = await db.scalar(
        select(Invoice).where(
            Invoice.id == invoice_id,
            Invoice.tenant_id == uuid.UUID(user.tenant_id),
        )
    )
    if not inv:
        raise HTTPException(404, "Invoice not found")
    try:
        delete_file(user.tenant_id, inv.storage_path)
    except Exception:
        pass
    await db.delete(inv)
    await db.commit()
    log.info("invoice.deleted", invoice_id=str(invoice_id), tenant_id=user.tenant_id)
    return {"data": {"deleted": True}, "error": None, "request_id": None}
