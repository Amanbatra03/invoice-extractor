import pytest
from unittest.mock import AsyncMock, MagicMock
from models.invoice import InvoiceSchema

@pytest.mark.asyncio
async def test_extraction_returns_invoice_schema():
    from agents.extraction_agent import run_extraction
    mock_retriever = AsyncMock()
    mock_retriever.all_chunks = AsyncMock(return_value=[
        {"text": "Vendor: Acme Corp\nTotal: $500.00\nInvoice #: INV-001", "page": 1}
    ])
    mock_provider = MagicMock()
    mock_provider.generate_structured.return_value = {
        "vendor_name": "Acme Corp", "invoice_number": "INV-001",
        "total_amount": 500.0, "currency": "USD",
        "invoice_date": None, "due_date": None, "subtotal": None,
        "tax": None, "po_number": None, "payment_terms": None,
        "vendor_tax_id": None, "vendor_address": None,
        "bill_to": None, "line_items": [],
    }
    result = await run_extraction(mock_retriever, mock_provider)
    assert isinstance(result, InvoiceSchema)
    assert result.vendor_name == "Acme Corp"
    assert result.total_amount == 500.0

@pytest.mark.asyncio
async def test_extraction_retries_on_schema_error():
    from agents.extraction_agent import run_extraction
    mock_retriever = AsyncMock()
    mock_retriever.all_chunks = AsyncMock(return_value=[
        {"text": "invoice content", "page": 1}
    ])
    mock_provider = MagicMock()
    mock_provider.generate_structured.side_effect = [
        ValueError("bad json"),
        {
            "vendor_name": "Retry Corp", "invoice_number": "INV-002",
            "total_amount": 100.0, "currency": "USD",
            "invoice_date": None, "due_date": None, "subtotal": None,
            "tax": None, "po_number": None, "payment_terms": None,
            "vendor_tax_id": None, "vendor_address": None,
            "bill_to": None, "line_items": [],
        }
    ]
    result = await run_extraction(mock_retriever, mock_provider)
    assert result.vendor_name == "Retry Corp"
    assert mock_provider.generate_structured.call_count == 2
