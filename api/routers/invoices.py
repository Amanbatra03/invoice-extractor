import asyncio
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query

from api.dependencies import get_current_user, require_roles, CurrentUser, get_queue
from api.schemas.invoice import InvoiceOut, InvoiceListResponse
from api.services.storage import upload_file, get_signed_url, delete_file, sha256_file
from api.supabase_client import get_service_client

router = APIRouter(prefix="/invoices", tags=["invoices"])
log = structlog.get_logger()

_ALLOWED_TYPES = {"application/pdf", "image/jpeg", "image/png", "image/jpg"}
_MAX_SIZE = 50 * 1024 * 1024  # 50 MB
_EXECUTABLE_MAGIC = (b"MZ", b"\x7fELF", b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe")


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
    if any(content.startswith(m) for m in _EXECUTABLE_MAGIC):
        raise HTTPException(400, "Executable files are not allowed")
    return _detect_file_type(file.filename or "", content)


async def _enqueue_ingest(invoice_id: uuid.UUID, job_id: uuid.UUID, queue) -> None:
    queue.enqueue(
        "workers.ingest_job.run",
        str(invoice_id),
        str(job_id),
        job_timeout=300,
    )


@router.post("/upload", response_model=dict)
async def upload_invoice(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(require_roles("admin", "analyst", "api_user")),
    queue=Depends(get_queue),
):
    content = await file.read()
    file_type = _validate_upload(file, content)
    sha = await asyncio.to_thread(sha256_file, content)
    fname = file.filename or f"upload.{file_type}"

    def _check_dup():
        c = get_service_client()
        return c.table("invoices").select("id").eq("tenant_id", user.tenant_id).eq("sha256", sha).limit(1).execute()

    dup = await asyncio.to_thread(_check_dup)
    if dup.data:
        existing_id = dup.data[0]["id"]
        return {
            "data": {"invoice_id": existing_id, "job_id": None, "status": "already_exists"},
            "error": None,
            "request_id": None,
        }

    storage_path = await asyncio.to_thread(upload_file, user.tenant_id, fname, content)
    now = datetime.now(timezone.utc).isoformat()
    uploaded_by = user.id if user.id and user.id != "api_key" else None

    def _insert_invoice():
        c = get_service_client()
        return c.table("invoices").insert({
            "tenant_id": user.tenant_id,
            "uploaded_by": uploaded_by,
            "file_name": fname,
            "file_type": file_type,
            "storage_path": storage_path,
            "sha256": sha,
            "status": "pending",
            "created_at": now,
        }).execute()

    inv_res = await asyncio.to_thread(_insert_invoice)
    invoice = inv_res.data[0]
    invoice_id = invoice["id"]

    def _insert_job():
        c = get_service_client()
        return c.table("jobs").insert({
            "tenant_id": user.tenant_id,
            "type": "ingest",
            "status": "queued",
            "payload": {"invoice_id": invoice_id},
            "created_at": now,
        }).execute()

    job_res = await asyncio.to_thread(_insert_job)
    job_id = job_res.data[0]["id"]

    await _enqueue_ingest(uuid.UUID(invoice_id), uuid.UUID(job_id), queue)
    log.info("invoice.uploaded", invoice_id=invoice_id, tenant_id=user.tenant_id)
    return {
        "data": {"invoice_id": invoice_id, "job_id": job_id, "status": "ingesting"},
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
):
    offset = (page - 1) * limit

    def _list():
        c = get_service_client()
        q = c.table("invoices").select("*", count="exact").eq("tenant_id", user.tenant_id)
        if status:
            q = q.eq("status", status)
        if file_type:
            q = q.eq("file_type", file_type)
        return q.range(offset, offset + limit - 1).execute()

    res = await asyncio.to_thread(_list)
    rows = res.data or []
    total = res.count or 0

    return {
        "data": InvoiceListResponse(
            items=[InvoiceOut.model_validate(r) for r in rows],
            total=total,
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
):
    def _get():
        c = get_service_client()
        return c.table("invoices").select("*").eq("id", str(invoice_id)).eq("tenant_id", user.tenant_id).limit(1).execute()

    res = await asyncio.to_thread(_get)
    if not res.data:
        raise HTTPException(404, "Invoice not found")
    return {"data": InvoiceOut.model_validate(res.data[0]), "error": None, "request_id": None}


@router.get("/{invoice_id}/download", response_model=dict)
async def download_invoice(
    invoice_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin", "analyst", "viewer")),
):
    def _get():
        c = get_service_client()
        return c.table("invoices").select("storage_path").eq("id", str(invoice_id)).eq("tenant_id", user.tenant_id).limit(1).execute()

    res = await asyncio.to_thread(_get)
    if not res.data:
        raise HTTPException(404, "Invoice not found")
    storage_path = res.data[0]["storage_path"]
    url = await asyncio.to_thread(get_signed_url, user.tenant_id, storage_path)
    return {"data": {"signed_url": url, "expires_in": 900}, "error": None, "request_id": None}


@router.delete("/{invoice_id}", response_model=dict)
async def delete_invoice(
    invoice_id: uuid.UUID,
    user: CurrentUser = Depends(require_roles("admin")),
):
    def _get():
        c = get_service_client()
        return c.table("invoices").select("id,storage_path").eq("id", str(invoice_id)).eq("tenant_id", user.tenant_id).limit(1).execute()

    res = await asyncio.to_thread(_get)
    if not res.data:
        raise HTTPException(404, "Invoice not found")
    storage_path = res.data[0]["storage_path"]

    try:
        await asyncio.to_thread(delete_file, user.tenant_id, storage_path)
    except Exception:
        pass

    def _delete():
        c = get_service_client()
        return c.table("invoices").delete().eq("id", str(invoice_id)).eq("tenant_id", user.tenant_id).execute()

    await asyncio.to_thread(_delete)
    log.info("invoice.deleted", invoice_id=str(invoice_id), tenant_id=user.tenant_id)
    return {"data": {"deleted": True}, "error": None, "request_id": None}
