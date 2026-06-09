import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

from models.invoice import InvoiceSchema
from rag.utils import extract_json_from_text

load_dotenv()

_SYSTEM_PROMPT = (
    "You are an expert invoice analyst. "
    "Answer questions using only information visible in the invoice image. "
    "If the requested information is not present, say 'Not found in invoice.'"
)

_EXTRACTION_PROMPT = """Extract all available invoice fields from this image.
Return ONLY a valid JSON object with these exact keys (use null for missing values):
{
  "vendor_name": string or null,
  "invoice_number": string or null,
  "invoice_date": string or null,
  "due_date": string or null,
  "subtotal": number or null,
  "tax": number or null,
  "total_amount": number or null,
  "currency": string or null,
  "line_items": [
    {"description": string, "quantity": number or null, "unit_price": number or null, "total": number or null}
  ]
}"""


def _get_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set. Add it to your .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def _validate_image(image_path: Path) -> Image.Image:
    try:
        img = Image.open(image_path)
        img.verify()
        return Image.open(image_path)
    except Exception as exc:
        raise ValueError(f"File does not appear to be a valid image: {exc}") from exc


def ask_invoice(image_path: Path, question: str) -> str:
    model = _get_model()
    img = _validate_image(image_path)
    response = model.generate_content([_SYSTEM_PROMPT, img, question])
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        raise RuntimeError(f"Gemini blocked this request: {response.prompt_feedback.block_reason}")
    return response.text


def extract_invoice_gemini(image_path: Path) -> InvoiceSchema:
    try:
        model = _get_model()
        img = _validate_image(image_path)
        response = model.generate_content([_EXTRACTION_PROMPT, img])
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return InvoiceSchema()
        json_str = extract_json_from_text(response.text)
        if json_str:
            return InvoiceSchema.model_validate_json(json_str)
    except Exception:
        pass
    return InvoiceSchema()
