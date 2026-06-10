import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from vision.gemini import ask_invoice, extract_invoice_gemini
from models.invoice import InvoiceSchema

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_mock_genai(text_response: str):
    mock_response = MagicMock()
    mock_response.text = text_response
    mock_response.prompt_feedback = None
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    return mock_model


def test_ask_invoice_returns_text(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"")

    with patch("vision.gemini._get_model") as mock_get, \
         patch("vision.gemini._validate_image") as mock_val:
        mock_get.return_value = _make_mock_genai("The total is $110.00")
        mock_val.return_value = MagicMock()

        result = ask_invoice(fake_image, "What is the total?")

    assert result == "The total is $110.00"


def test_ask_invoice_raises_on_gemini_block(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"")

    mock_feedback = MagicMock()
    mock_feedback.block_reason = "SAFETY"
    mock_response = MagicMock()
    mock_response.text = ""
    mock_response.prompt_feedback = mock_feedback
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    with patch("vision.gemini._get_model", return_value=mock_model), \
         patch("vision.gemini._validate_image") as mock_val:
        mock_val.return_value = MagicMock()
        with pytest.raises(RuntimeError, match="Gemini blocked"):
            ask_invoice(fake_image, "What is the total?")


def test_extract_invoice_gemini_returns_schema(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"")

    json_resp = (
        '{"vendor_name": "Test Corp", "invoice_number": "T-001", '
        '"invoice_date": null, "due_date": null, "subtotal": null, '
        '"tax": null, "total_amount": 99.0, "currency": "USD", "line_items": []}'
    )

    with patch("vision.gemini._get_model") as mock_get, \
         patch("vision.gemini._validate_image") as mock_val:
        mock_get.return_value = _make_mock_genai(json_resp)
        mock_val.return_value = MagicMock()

        result = extract_invoice_gemini(fake_image)

    assert isinstance(result, InvoiceSchema)
    assert result.vendor_name == "Test Corp"
    assert result.total_amount == 99.0


def test_extract_invoice_gemini_raises_without_api_key(tmp_path, monkeypatch):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
        extract_invoice_gemini(fake_image)


def test_extract_invoice_gemini_returns_empty_on_bad_response(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"")

    with patch("vision.gemini._get_model") as mock_get, \
         patch("vision.gemini._validate_image") as mock_val:
        mock_get.return_value = _make_mock_genai("Sorry, cannot extract.")
        mock_val.return_value = MagicMock()

        result = extract_invoice_gemini(fake_image)

    assert isinstance(result, InvoiceSchema)
    assert result.vendor_name is None
