import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from PIL import Image

from models.invoice import InvoiceSchema
from rag.utils import extract_json_from_text

load_dotenv()

_MODEL = "gemini-2.0-flash"

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


def _get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set. Add it to your .env file.")
    return genai.Client(api_key=api_key)


def _validate_image(image_path: Path) -> Image.Image:
    try:
        img = Image.open(image_path)
        img.verify()
        return Image.open(image_path)
    except Exception as exc:
        raise ValueError(f"File does not appear to be a valid image: {exc}") from exc


def ask_invoice(image_path: Path, question: str) -> str:
    client = _get_client()
    img = _validate_image(image_path)
    response = client.models.generate_content(
        model=_MODEL, contents=[_SYSTEM_PROMPT, img, question]
    )
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        raise RuntimeError(f"Gemini blocked this request: {response.prompt_feedback.block_reason}")
    return response.text


def extract_invoice_gemini(image_path: Path) -> InvoiceSchema:
    # Config/input errors must propagate so the UI can surface them;
    # only model-output problems degrade to an empty schema.
    client = _get_client()
    img = _validate_image(image_path)
    try:
        response = client.models.generate_content(
            model=_MODEL, contents=[_EXTRACTION_PROMPT, img]
        )
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return InvoiceSchema()
        json_str = extract_json_from_text(response.text)
        if json_str:
            return InvoiceSchema.model_validate_json(json_str)
    except Exception:
        pass
    return InvoiceSchema()
