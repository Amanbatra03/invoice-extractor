import pytest
from models.invoice import InvoiceSchema, LineItem
from agents.validation_agent import run_validation, ValidationReport

def test_arithmetic_check_catches_mismatch():
    schema = InvoiceSchema(
        vendor_name="Acme", subtotal=100.0, tax=10.0, total_amount=120.0
    )
    report = run_validation(schema)
    issues = [i for i in report.issues if i["check"] == "arithmetic"]
    assert len(issues) == 1
    assert issues[0]["severity"] == "warning"

def test_arithmetic_check_passes_on_correct():
    schema = InvoiceSchema(
        vendor_name="Acme", subtotal=100.0, tax=10.0, total_amount=110.0
    )
    report = run_validation(schema)
    issues = [i for i in report.issues if i["check"] == "arithmetic"]
    assert len(issues) == 0

def test_line_item_sum_check():
    schema = InvoiceSchema(
        subtotal=100.0,
        line_items=[
            LineItem(description="A", total=50.0),
            LineItem(description="B", total=40.0),
        ]
    )
    report = run_validation(schema)
    issues = [i for i in report.issues if i["check"] == "line_item_sum"]
    assert len(issues) == 1

def test_future_date_check():
    schema = InvoiceSchema(invoice_date="2099-01-01")
    report = run_validation(schema)
    issues = [i for i in report.issues if i["check"] == "future_date"]
    assert len(issues) == 1

def test_no_issues_on_clean_invoice():
    schema = InvoiceSchema(
        vendor_name="Acme", invoice_number="INV-001",
        subtotal=100.0, tax=10.0, total_amount=110.0,
        invoice_date="2026-01-15",
        line_items=[LineItem(description="Service", total=100.0)],
    )
    report = run_validation(schema)
    arithmetic_issues = [i for i in report.issues if i["check"] == "arithmetic"]
    future_issues = [i for i in report.issues if i["check"] == "future_date"]
    assert len(arithmetic_issues) == 0
    assert len(future_issues) == 0
    assert report.passed
