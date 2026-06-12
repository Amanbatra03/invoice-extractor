from rag.hybrid_retriever import HybridRetriever
from rag.utils import ExtractionError, extract_json_from_text
from models.invoice import InvoiceSchema

_EXTRACTION_PROMPT = """You are an invoice data extractor. Extract all fields from the invoice context below.
Return ONLY a valid JSON object with these exact keys (use null for missing values):
{{
  "vendor_name": string or null,
  "invoice_number": string or null,
  "invoice_date": string or null,
  "due_date": string or null,
  "subtotal": number or null,
  "tax": number or null,
  "total_amount": number or null,
  "currency": string or null,
  "po_number": string or null,
  "payment_terms": string or null,
  "vendor_tax_id": string or null,
  "vendor_address": string or null,
  "bill_to": string or null,
  "line_items": [
    {{"description": string, "quantity": number or null, "unit_price": number or null, "total": number or null}}
  ]
}}

Invoice context:
{context}"""


def extract_invoice(retriever: HybridRetriever, llm) -> InvoiceSchema:
    chunks = retriever.all_chunks()
    context = "\n\n".join(c["text"] for c in chunks)
    raw = llm.invoke(_EXTRACTION_PROMPT.format(context=context))
    json_str = extract_json_from_text(raw)
    if json_str is None:
        raise ExtractionError("Model response contained no JSON object.")
    try:
        return InvoiceSchema.model_validate_json(json_str)
    except Exception as exc:
        raise ExtractionError(f"Model JSON did not match the invoice schema: {exc}") from exc
