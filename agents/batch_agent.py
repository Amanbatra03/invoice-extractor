import asyncio
import uuid as _uuid
from models.invoice import InvoiceSchema
from agents.base import LLMProvider
from agents.extraction_agent import run_extraction
from agents.retriever import HybridRetriever


async def _extract_single(
    invoice_id: str,
    db,
    provider: LLMProvider,
) -> InvoiceSchema:
    retriever = HybridRetriever(
        invoice_id=_uuid.UUID(invoice_id),
        db=db,
        provider=provider,
    )
    return await run_extraction(retriever, provider)


async def run_batch(
    invoice_ids: list[str],
    db,
    provider: LLMProvider,
    max_concurrent: int = 5,
) -> dict:
    done: dict[str, InvoiceSchema] = {}
    failed: dict[str, str] = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process(invoice_id: str) -> None:
        async with semaphore:
            try:
                schema = await _extract_single(invoice_id, db, provider)
                done[invoice_id] = schema
            except Exception as exc:
                failed[invoice_id] = str(exc)

    await asyncio.gather(*[_process(inv_id) for inv_id in invoice_ids])
    return {
        "done": done,
        "failed": failed,
        "total": len(invoice_ids),
        "success_count": len(done),
        "failure_count": len(failed),
    }
