from models.invoice import InvoiceSchema
from agents.base import LLMProvider
from agents.retriever import HybridRetriever

_EXTRACTION_PROMPT = """You are an invoice data extractor.
Extract all available fields from the invoice text below.
Return a complete InvoiceSchema JSON object.

Invoice text:
{context}"""

_RETRY_PROMPT = """Previous extraction attempt failed with error: {error}
Please re-extract the invoice data carefully, ensuring the JSON is valid.

Invoice text:
{context}"""

MAX_RETRIES = 2


async def run_extraction(
    retriever: HybridRetriever,
    provider: LLMProvider,
) -> InvoiceSchema:
    chunks = await retriever.all_chunks()
    context = "\n\n".join(c["text"] for c in chunks)
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            if attempt == 0:
                prompt = _EXTRACTION_PROMPT.format(context=context)
            else:
                prompt = _RETRY_PROMPT.format(error=str(last_error), context=context)
            raw = provider.generate_structured(prompt, InvoiceSchema)
            return InvoiceSchema.model_validate(raw)
        except Exception as exc:
            last_error = exc
            if attempt == MAX_RETRIES:
                raise ValueError(
                    f"Extraction failed after {MAX_RETRIES + 1} attempts: {exc}"
                ) from exc
