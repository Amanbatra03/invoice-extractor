from models.invoice import InvoiceSchema

_HEADER_FIELDS = [
    "vendor_name", "invoice_number", "invoice_date", "subtotal", "tax",
    "total_amount", "currency", "po_number", "payment_terms", "vendor_tax_id",
]


def field_match(expected, actual) -> bool:
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= max(abs(float(expected)) * 0.01, 0.011)
    return str(expected).strip().lower() == str(actual).strip().lower()


def score_invoice(truth: InvoiceSchema, predicted: InvoiceSchema) -> dict[str, float]:
    scores = {
        f: 1.0 if field_match(getattr(truth, f), getattr(predicted, f)) else 0.0
        for f in _HEADER_FIELDS
    }
    if truth.line_items:
        n_expected = len(truth.line_items)
        matched = 0
        remaining = list(predicted.line_items)
        for t in truth.line_items:
            for p in remaining:
                if field_match(t.total, p.total) and field_match(t.quantity, p.quantity):
                    matched += 1
                    remaining.remove(p)
                    break
        scores["line_items"] = matched / n_expected
    return scores
