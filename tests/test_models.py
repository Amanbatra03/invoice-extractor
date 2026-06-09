import pytest
from models.invoice import InvoiceSchema, LineItem


def test_invoice_schema_full():
    data = {
        "vendor_name": "ACME Corp",
        "invoice_number": "INV-001",
        "invoice_date": "2024-01-15",
        "due_date": "2024-02-15",
        "subtotal": 100.0,
        "tax": 10.0,
        "total_amount": 110.0,
        "currency": "USD",
        "line_items": [
            {"description": "Widget", "quantity": 2.0, "unit_price": 50.0, "total": 100.0}
        ],
    }
    schema = InvoiceSchema(**data)
    assert schema.vendor_name == "ACME Corp"
    assert schema.total_amount == 110.0
    assert len(schema.line_items) == 1
    assert schema.line_items[0].description == "Widget"


def test_invoice_schema_all_none():
    schema = InvoiceSchema()
    assert schema.vendor_name is None
    assert schema.total_amount is None
    assert schema.line_items == []


def test_line_item_description_only():
    item = LineItem(description="Service Fee")
    assert item.description == "Service Fee"
    assert item.quantity is None
    assert item.unit_price is None
    assert item.total is None


def test_invoice_schema_from_json():
    json_str = '{"vendor_name": "Corp A", "total_amount": 55.5, "line_items": []}'
    schema = InvoiceSchema.model_validate_json(json_str)
    assert schema.vendor_name == "Corp A"
    assert schema.total_amount == 55.5
