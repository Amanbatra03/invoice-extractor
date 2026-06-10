import pytest
from unittest.mock import MagicMock
from rag.extractor import extract_invoice, ExtractionError
from models.invoice import InvoiceSchema


def _make_retriever(text: str):
    r = MagicMock()
    r.all_chunks.return_value = [{"text": text, "page": 1}]
    r.retrieve.return_value = [{"text": text, "page": 1, "score": 0.9}]
    return r


def test_extract_uses_whole_document_not_retrieval():
    retriever = _make_retriever("Total: $110.00")
    llm = MagicMock()
    llm.invoke.return_value = '{"vendor_name": "X", "line_items": []}'
    extract_invoice(retriever, llm)
    retriever.all_chunks.assert_called_once()
    retriever.retrieve.assert_not_called()


def test_extract_returns_invoice_schema():
    retriever = _make_retriever("Total: $110.00, Vendor: ACME Corp, Invoice #INV-001")
    llm = MagicMock()
    llm.invoke.return_value = (
        '{"vendor_name": "ACME Corp", "invoice_number": "INV-001", '
        '"invoice_date": null, "due_date": null, "subtotal": null, '
        '"tax": null, "total_amount": 110.0, "currency": "USD", "line_items": []}'
    )
    result = extract_invoice(retriever, llm)
    assert isinstance(result, InvoiceSchema)
    assert result.vendor_name == "ACME Corp"
    assert result.total_amount == 110.0


def test_extract_raises_on_bad_json():
    retriever = _make_retriever("some invoice text")
    llm = MagicMock()
    llm.invoke.return_value = "I cannot extract the fields."
    with pytest.raises(ExtractionError):
        extract_invoice(retriever, llm)


def test_extract_raises_on_invalid_schema():
    retriever = _make_retriever("some invoice text")
    llm = MagicMock()
    llm.invoke.return_value = '{"vendor_name": {"nested": "wrong type"}}'
    with pytest.raises(ExtractionError):
        extract_invoice(retriever, llm)
