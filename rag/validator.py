from models.invoice import InvoiceSchema

_REL_TOLERANCE = 0.02   # 2% relative slack for OCR/LLM rounding
_ABS_TOLERANCE = 0.05   # at least 5 cents absolute


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(abs(b) * _REL_TOLERANCE, _ABS_TOLERANCE)


def validate_invoice(schema: InvoiceSchema) -> list[str]:
    """Deterministic arithmetic checks; returns human-readable warnings."""
    warnings: list[str] = []

    for i, li in enumerate(schema.line_items, 1):
        if li.quantity is not None and li.unit_price is not None and li.total is not None:
            expected = li.quantity * li.unit_price
            if not _close(expected, li.total):
                warnings.append(
                    f"Line {i} ({li.description[:40]}): qty × unit price = "
                    f"{expected:.2f}, but line total is {li.total:.2f}"
                )

    item_totals = [li.total for li in schema.line_items if li.total is not None]
    if item_totals and schema.subtotal is not None and not _close(sum(item_totals), schema.subtotal):
        warnings.append(
            f"Line items sum to {sum(item_totals):.2f}, but subtotal is {schema.subtotal:.2f}"
        )

    if schema.subtotal is not None and schema.tax is not None and schema.total_amount is not None:
        expected = schema.subtotal + schema.tax
        if not _close(expected, schema.total_amount):
            warnings.append(
                f"Subtotal + tax = {expected:.2f}, but total is {schema.total_amount:.2f}"
            )

    for field in ("subtotal", "tax", "total_amount"):
        value = getattr(schema, field)
        if value is not None and value < 0:
            warnings.append(f"{field} is negative: {value}")

    return warnings


def has_amounts(schema: InvoiceSchema) -> bool:
    """True if the extraction contains at least one monetary value."""
    if any(getattr(schema, f) is not None for f in ("subtotal", "tax", "total_amount")):
        return True
    return any(li.total is not None or li.unit_price is not None for li in schema.line_items)
