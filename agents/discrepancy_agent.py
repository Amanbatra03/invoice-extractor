from datetime import datetime
from models.invoice import InvoiceSchema
from agents.base import LLMProvider

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


def _semantic_vendor_check(
    vendor_a: str, vendor_b: str, provider: LLMProvider
) -> bool:
    """Returns True if Gemini thinks the vendors are the same entity."""
    prompt = (
        f'Are "{vendor_a}" and "{vendor_b}" the same company or legal entity? '
        f"Reply ONLY 'yes' or 'no'."
    )
    verdict = provider.generate(prompt).strip().lower()
    return verdict.startswith("yes")


def run_comparison(
    named_schemas: list[tuple[str, InvoiceSchema]],
    provider: LLMProvider,
) -> dict:
    if len(named_schemas) < 2:
        return {"table": {}, "discrepancies": []}

    table = {
        field: {name: getattr(schema, field) for name, schema in named_schemas}
        for field in _FIELDS
    }
    discrepancies: list[dict] = []

    # Phase 1 — deterministic
    vendors = [v.strip() for v in table["vendor_name"].values() if v and v.strip()]
    unique_vendors = {v.lower() for v in vendors}
    if len(unique_vendors) > 1:
        # Phase 2 — Gemini semantic check for vendor aliases
        vendor_list = list(vendors)
        is_same = _semantic_vendor_check(vendor_list[0], vendor_list[1], provider)
        if not is_same:
            discrepancies.append({
                "field": "vendor_name",
                "severity": "critical",
                "detail": f"Different vendors: {', '.join(sorted(set(vendors)))}",
                "ai_reasoning": "Gemini confirmed these are different entities",
            })

    currencies = {c.strip().upper() for c in table["currency"].values() if c and c.strip()}
    if len(currencies) > 1:
        discrepancies.append({
            "field": "currency",
            "severity": "warning",
            "detail": f"Mixed currencies — totals not comparable: {', '.join(sorted(currencies))}",
            "ai_reasoning": None,
        })

    totals = [(name, val) for name, val in table["total_amount"].items() if val is not None]
    if len(totals) >= 2 and len(currencies) <= 1:
        amounts = [v for _, v in totals]
        min_a, max_a = min(amounts), max(amounts)
        if min_a > 0 and (max_a - min_a) / min_a > 0.05:
            discrepancies.append({
                "field": "total_amount",
                "severity": "warning",
                "detail": f"Total mismatch >5%: {[f'{n}={v}' for n, v in totals]}",
                "ai_reasoning": None,
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
                "severity": "info",
                "detail": f"Date gap of {gap} days between invoices",
                "ai_reasoning": None,
            })

    inv_numbers = [v for v in table["invoice_number"].values() if v]
    if len(inv_numbers) == len(named_schemas) and len(set(inv_numbers)) == 1:
        discrepancies.append({
            "field": "invoice_number",
            "severity": "critical",
            "detail": f"Duplicate invoice number: {inv_numbers[0]}",
            "ai_reasoning": None,
        })

    return {"table": table, "discrepancies": discrepancies}
