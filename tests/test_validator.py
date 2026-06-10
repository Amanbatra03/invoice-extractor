from models.invoice import InvoiceSchema, LineItem
from rag.validator import validate_invoice, has_amounts


def test_consistent_invoice_has_no_warnings():
    schema = InvoiceSchema(
        subtotal=100.0, tax=10.0, total_amount=110.0,
        line_items=[LineItem(description="A", quantity=2, unit_price=25.0, total=50.0),
                    LineItem(description="B", quantity=1, unit_price=50.0, total=50.0)],
    )
    assert validate_invoice(schema) == []


def test_line_item_arithmetic_mismatch_flagged():
    schema = InvoiceSchema(
        line_items=[LineItem(description="Widget", quantity=2, unit_price=25.0, total=99.0)],
    )
    warnings = validate_invoice(schema)
    assert len(warnings) == 1
    assert "Widget" in warnings[0]


def test_subtotal_vs_items_mismatch_flagged():
    schema = InvoiceSchema(
        subtotal=500.0,
        line_items=[LineItem(description="A", total=50.0), LineItem(description="B", total=50.0)],
    )
    warnings = validate_invoice(schema)
    assert any("subtotal" in w.lower() for w in warnings)


def test_total_vs_subtotal_plus_tax_mismatch_flagged():
    schema = InvoiceSchema(subtotal=100.0, tax=10.0, total_amount=999.0)
    warnings = validate_invoice(schema)
    assert any("total" in w.lower() for w in warnings)


def test_negative_amount_flagged():
    schema = InvoiceSchema(total_amount=-5.0)
    warnings = validate_invoice(schema)
    assert any("negative" in w.lower() for w in warnings)


def test_missing_fields_produce_no_warnings():
    assert validate_invoice(InvoiceSchema()) == []


def test_small_rounding_differences_tolerated():
    schema = InvoiceSchema(subtotal=192.81, tax=19.28, total_amount=212.09)
    assert validate_invoice(schema) == []


def test_has_amounts_true_with_total():
    assert has_amounts(InvoiceSchema(total_amount=10.0)) is True


def test_has_amounts_true_with_line_item_total():
    assert has_amounts(InvoiceSchema(line_items=[LineItem(description="A", total=5.0)])) is True


def test_has_amounts_false_when_empty():
    assert has_amounts(InvoiceSchema()) is False
