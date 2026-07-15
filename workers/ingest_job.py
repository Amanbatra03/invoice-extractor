import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import select, update

from agents.base import get_provider
from api.config import get_settings
from api.services.storage import download_file
from db.models import Invoice, InvoiceChunk, Job
from db.session import get_session_factory

log = structlog.get_logger()


def _extract_page_texts(pdf_bytes: bytes) -> list[str]:
    import tempfile
    from pypdf import PdfReader
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)
    try:
        reader = PdfReader(str(tmp_path))
        texts = [p.extract_text() or "" for p in reader.pages]
        if sum(len(t.strip()) for t in texts) < 32:
            import rag.ocr
            texts = rag.ocr.ocr_pdf_pages(tmp_path)
        return texts
    finally:
        tmp_path.unlink(missing_ok=True)


def _chunk_pdf_bytes(pdf_bytes: bytes, chunk_size: int = 800, chunk_overlap: int = 80) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    page_texts = _extract_page_texts(pdf_bytes)
    chunks = []
    for page_num, page_text in enumerate(page_texts, start=1):
        for piece in splitter.split_text(page_text):
            chunks.append({"text": piece, "page": page_num})
    return chunks


async def _store_chunks(
    chunks: list[dict],
    invoice_id: uuid.UUID,
    tenant_id: uuid.UUID,
    db,
    provider,
) -> None:
    texts = [c["text"] for c in chunks]
    embeddings = provider.embed_text(texts)
    for chunk, embedding in zip(chunks, embeddings):
        db.add(InvoiceChunk(
            invoice_id=invoice_id,
            tenant_id=tenant_id,
            chunk_text=chunk["text"],
            page_num=chunk["page"],
            embedding=embedding,
            created_at=datetime.now(timezone.utc),
        ))
    await db.commit()


def run(invoice_id_str: str, job_id_str: str) -> None:
    import asyncio
    asyncio.run(_run_async(invoice_id_str, job_id_str))


async def _run_async(invoice_id_str: str, job_id_str: str) -> None:
    settings = get_settings()
    invoice_id = uuid.UUID(invoice_id_str)
    job_id = uuid.UUID(job_id_str)
    session_factory = get_session_factory()

    async with session_factory() as db:
        inv = await db.scalar(select(Invoice).where(Invoice.id == invoice_id))
        if not inv:
            log.error("ingest.invoice_not_found", invoice_id=invoice_id_str)
            return

        await db.execute(
            update(Invoice).where(Invoice.id == invoice_id).values(status="ingesting")
        )
        await db.execute(
            update(Job).where(Job.id == job_id).values(status="running")
        )
        await db.commit()

        try:
            pdf_bytes = await asyncio.to_thread(download_file, str(inv.tenant_id), inv.storage_path)
            if inv.file_type == "pdf":
                chunks = await asyncio.to_thread(
                    _chunk_pdf_bytes,
                    pdf_bytes,
                    getattr(settings, "CHUNK_SIZE", 800),
                    getattr(settings, "CHUNK_OVERLAP", 80),
                )
            else:
                import tempfile
                suffix = "." + inv.file_name.rsplit(".", 1)[-1]
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = Path(tmp.name)
                provider = get_provider()
                embedding = provider.embed_image(tmp_path)
                tmp_path.unlink(missing_ok=True)
                chunks = [{"text": f"[image invoice: {inv.file_name}]", "page": 1, "_embedding": embedding}]

            provider = get_provider()
            await _store_chunks(chunks, invoice_id, inv.tenant_id, db, provider)

            await db.execute(
                update(Invoice).where(Invoice.id == invoice_id).values(status="ready")
            )
            await db.execute(
                update(Job).where(Job.id == job_id).values(
                    status="done",
                    result={"chunks_stored": len(chunks)},
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()
            log.info("ingest.done", invoice_id=invoice_id_str, chunks=len(chunks))

        except Exception as exc:
            log.error("ingest.failed", invoice_id=invoice_id_str, error=str(exc))
            await db.execute(
                update(Invoice).where(Invoice.id == invoice_id).values(status="failed")
            )
            await db.execute(
                update(Job).where(Job.id == job_id).values(
                    status="failed",
                    error=str(exc),
                    completed_at=datetime.now(timezone.utc),
                )
            )
            await db.commit()
            raise
