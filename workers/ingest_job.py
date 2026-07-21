import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog

from api.services.storage import download_file
from api.supabase_client import get_service_client

log = structlog.get_logger()


def _fetch_invoice(invoice_id: str) -> dict | None:
    c = get_service_client()
    res = c.table("invoices").select("*").eq("id", invoice_id).limit(1).execute()
    return res.data[0] if res.data else None


def _set_invoice(invoice_id: str, **fields) -> None:
    c = get_service_client()
    c.table("invoices").update(fields).eq("id", invoice_id).execute()


def _set_job(job_id: str, **fields) -> None:
    c = get_service_client()
    c.table("jobs").update(fields).eq("id", job_id).execute()


def _chunk_pdf_bytes(pdf_bytes: bytes, chunk_size: int = 800, chunk_overlap: int = 80) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader

    with Path("/tmp/ingest_tmp.pdf").open("wb") as fh:
        fh.write(pdf_bytes)
    tmp_path = Path("/tmp/ingest_tmp.pdf")
    try:
        reader = PdfReader(str(tmp_path))
        page_texts = [p.extract_text() or "" for p in reader.pages]
        if sum(len(t.strip()) for t in page_texts) < 32:
            import rag.ocr
            page_texts = rag.ocr.ocr_pdf_pages(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for page_num, page_text in enumerate(page_texts, start=1):
        for piece in splitter.split_text(page_text):
            chunks.append({"text": piece, "page": page_num})
    return chunks


def _store_chunks(chunks: list[dict], invoice_id: str, tenant_id: str, provider) -> int:
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks]
    embeddings = provider.embed_text(texts)
    now = datetime.now(timezone.utc).isoformat()
    records = [
        {
            "invoice_id": invoice_id,
            "tenant_id": tenant_id,
            "chunk_text": chunk["text"],
            "page_num": chunk["page"],
            "embedding": list(emb),
            "created_at": now,
        }
        for chunk, emb in zip(chunks, embeddings)
    ]
    c = get_service_client()
    c.table("invoice_chunks").insert(records).execute()
    return len(records)


def _run_sync(invoice_id_str: str, job_id_str: str) -> None:
    from api.config import get_settings
    settings = get_settings()
    now = datetime.now(timezone.utc).isoformat()

    inv = _fetch_invoice(invoice_id_str)
    if not inv:
        log.error("ingest.invoice_not_found", invoice_id=invoice_id_str)
        return

    _set_invoice(invoice_id_str, status="ingesting")
    _set_job(job_id_str, status="running")

    try:
        pdf_bytes = download_file(inv["tenant_id"], inv["storage_path"])

        if inv["file_type"] == "pdf":
            chunks = _chunk_pdf_bytes(
                pdf_bytes,
                getattr(settings, "CHUNK_SIZE", 800),
                getattr(settings, "CHUNK_OVERLAP", 80),
            )
        else:
            chunks = [{"text": f"[image invoice: {inv['file_name']}]", "page": 1}]

        from agents.base import get_provider
        provider = get_provider()
        n_chunks = _store_chunks(chunks, invoice_id_str, inv["tenant_id"], provider)

        _set_invoice(invoice_id_str, status="ready")
        _set_job(job_id_str, status="completed", result={"chunks_stored": n_chunks}, completed_at=now)
        log.info("ingest.done", invoice_id=invoice_id_str, chunks=n_chunks)

    except Exception as exc:
        log.error("ingest.failed", invoice_id=invoice_id_str, error=str(exc))
        _set_invoice(invoice_id_str, status="failed")
        _set_job(job_id_str, status="failed", error=str(exc)[:2000], completed_at=now)


async def run_bg(invoice_id_str: str, job_id_str: str) -> None:
    """Entry point for FastAPI BackgroundTasks — runs sync body in thread pool."""
    await asyncio.to_thread(_run_sync, invoice_id_str, job_id_str)


def run(invoice_id_str: str, job_id_str: str) -> None:
    """Legacy RQ entry point — kept for compatibility."""
    asyncio.run(run_bg(invoice_id_str, job_id_str))
