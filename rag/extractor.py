from rag.hybrid_retriever import HybridRetriever
from rag.utils import extract_json_from_text
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
  "line_items": [
    {{"description": string, "quantity": number or null, "unit_price": number or null, "total": number or null}}
  ]
}}

Invoice context:
{context}"""


def extract_invoice(
    retriever: HybridRetriever, llm, query: str = "extract all invoice fields"
) -> InvoiceSchema:
    chunks = retriever.retrieve(query)
    context = "\n\n".join(c["text"] for c in chunks)
    raw = llm.invoke(_EXTRACTION_PROMPT.format(context=context))
    json_str = extract_json_from_text(raw)
    if json_str is None:
        return InvoiceSchema()
    try:
        return InvoiceSchema.model_validate_json(json_str)
    except Exception:
        return InvoiceSchema()
