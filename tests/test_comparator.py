import pytest
from models.invoice import InvoiceSchema
from rag.comparator import compare_invoices


def _schema(**kwargs) -> InvoiceSchema:
    return InvoiceSchema(**kwargs)


def test_compare_identical_invoices_no_discrepancies():
    s = _schema(vendor_name="ACME", total_amount=100.0, invoice_date="2024-01-15")
    result = compare_invoices([("inv1", s), ("inv2", s)])
    assert result["discrepancies"] == []


def test_compare_different_vendors_flagged():
    a = _schema(vendor_name="ACME Corp")
    b = _schema(vendor_name="Beta Ltd")
    result = compare_invoices([("a", a), ("b", b)])
    fields = [d["field"] for d in result["discrepancies"]]
    assert "vendor_name" in fields


def test_compare_total_mismatch_over_5pct_flagged():
    a = _schema(total_amount=100.0)
    b = _schema(total_amount=110.0)
    result = compare_invoices([("a", a), ("b", b)])
    fields = [d["field"] for d in result["discrepancies"]]
    assert "total_amount" in fields


def test_compare_total_mismatch_under_5pct_not_flagged():
    a = _schema(total_amount=100.0)
    b = _schema(total_amount=102.0)
    result = compare_invoices([("a", a), ("b", b)])
    fields = [d["field"] for d in result["discrepancies"]]
    assert "total_amount" not in fields


def test_compare_date_gap_over_30_days_flagged():
    a = _schema(invoice_date="2024-01-01")
    b = _schema(invoice_date="2024-02-15")
    result = compare_invoices([("a", a), ("b", b)])
    fields = [d["field"] for d in result["discrepancies"]]
    assert "invoice_date" in fields


def test_compare_date_gap_under_30_days_not_flagged():
    a = _schema(invoice_date="2024-01-01")
    b = _schema(invoice_date="2024-01-20")
    result = compare_invoices([("a", a), ("b", b)])
    fields = [d["field"] for d in result["discrepancies"]]
    assert "invoice_date" not in fields


def test_compare_returns_table_with_all_fields():
    a = _schema(vendor_name="ACME", total_amount=100.0)
    b = _schema(vendor_name="ACME", total_amount=100.0)
    result = compare_invoices([("a", a), ("b", b)])
    assert "vendor_name" in result["table"]
    assert result["table"]["vendor_name"] == {"a": "ACME", "b": "ACME"}


def test_compare_single_invoice_returns_empty():
    s = _schema(vendor_name="ACME")
    result = compare_invoices([("only", s)])
    assert result == {"table": {}, "discrepancies": []}
