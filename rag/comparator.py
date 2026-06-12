from datetime import datetime
from models.invoice import InvoiceSchema

_FIELDS = [
    "vendor_name", "invoice_number", "invoice_date", "due_date",
    "subtotal", "tax", "total_amount", "currency", "po_number",
]

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y")


def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def compare_invoices(named_schemas: list[tuple[str, InvoiceSchema]]) -> dict:
    if len(named_schemas) < 2:
        return {"table": {}, "discrepancies": []}

    table = {
        field: {name: getattr(schema, field) for name, schema in named_schemas}
        for field in _FIELDS
    }

    discrepancies: list[dict] = []

    vendors = [v.strip() for v in table["vendor_name"].values() if v and v.strip()]
    if len({v.lower() for v in vendors}) > 1:
        discrepancies.append({
            "field": "vendor_name",
            "detail": f"Different vendors: {', '.join(sorted(set(vendors)))}",
        })

    currencies = {c.strip().upper() for c in table["currency"].values() if c and c.strip()}
    if len(currencies) > 1:
        discrepancies.append({
            "field": "currency",
            "detail": f"Mixed currencies — totals not compared: {', '.join(sorted(currencies))}",
        })

    totals = [(name, val) for name, val in table["total_amount"].items() if val is not None]
    if len(totals) >= 2 and len(currencies) <= 1:
        amounts = [v for _, v in totals]
        min_a, max_a = min(amounts), max(amounts)
        if min_a > 0 and (max_a - min_a) / min_a > 0.05:
            discrepancies.append({
                "field": "total_amount",
                "detail": f"Total mismatch >5%: {[f'{n}={v}' for n, v in totals]}",
            })

    parsed_dates = [
        (name, d)
        for name, val in table["invoice_date"].items()
        if (d := _parse_date(val)) is not None
    ]
    if len(parsed_dates) >= 2:
        date_values = [d for _, d in parsed_dates]
        gap = (max(date_values) - min(date_values)).days
        if gap > 30:
            discrepancies.append({
                "field": "invoice_date",
                "detail": f"Date gap of {gap} days between invoices",
            })

    return {"table": table, "discrepancies": discrepancies}
