import pytest
from unittest.mock import MagicMock
from models.invoice import InvoiceSchema

def test_detects_total_mismatch():
    from agents.discrepancy_agent import run_comparison
    schemas = [
        ("inv_a.pdf", InvoiceSchema(vendor_name="Acme", total_amount=1000.0, currency="USD")),
        ("inv_b.pdf", InvoiceSchema(vendor_name="Acme", total_amount=1200.0, currency="USD")),
    ]
    mock_provider = MagicMock()
    mock_provider.generate.return_value = "no"
    result = run_comparison(schemas, mock_provider)
    fields = [d["field"] for d in result["discrepancies"]]
    assert "total_amount" in fields

def test_detects_currency_mismatch():
    from agents.discrepancy_agent import run_comparison
    schemas = [
        ("inv_a.pdf", InvoiceSchema(vendor_name="Acme", total_amount=100.0, currency="USD")),
        ("inv_b.pdf", InvoiceSchema(vendor_name="Acme", total_amount=100.0, currency="EUR")),
    ]
    mock_provider = MagicMock()
    result = run_comparison(schemas, mock_provider)
    fields = [d["field"] for d in result["discrepancies"]]
    assert "currency" in fields

def test_no_discrepancies_on_matching_invoices():
    from agents.discrepancy_agent import run_comparison
    schemas = [
        ("inv_a.pdf", InvoiceSchema(vendor_name="Acme", total_amount=500.0, currency="USD")),
        ("inv_b.pdf", InvoiceSchema(vendor_name="Acme", total_amount=500.0, currency="USD")),
    ]
    mock_provider = MagicMock()
    result = run_comparison(schemas, mock_provider)
    assert result["discrepancies"] == []
