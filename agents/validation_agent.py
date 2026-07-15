from dataclasses import dataclass, field
from datetime import datetime, timezone

from models.invoice import InvoiceSchema

_TOLERANCE = 0.02


@dataclass
class ValidationReport:
    issues: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i["severity"] in ("warning", "critical") for i in self.issues)

    def add(self, check: str, severity: str, detail: str) -> None:
        self.issues.append({"check": check, "severity": severity, "detail": detail})


def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def run_validation(schema: InvoiceSchema) -> ValidationReport:
    report = ValidationReport()
    now = datetime.now(timezone.utc)

    # Arithmetic: subtotal + tax = total
    if all(v is not None for v in [schema.subtotal, schema.tax, schema.total_amount]):
        expected = round(schema.subtotal + schema.tax, 2)
        actual = round(schema.total_amount, 2)
        if abs(expected - actual) > _TOLERANCE:
            report.add(
                "arithmetic", "warning",
                f"subtotal ({schema.subtotal}) + tax ({schema.tax}) = {expected}, "
                f"but total_amount = {actual}",
            )

    # Line item sum vs subtotal
    if schema.line_items and schema.subtotal is not None:
        totals = [li.total for li in schema.line_items if li.total is not None]
        if totals:
            items_sum = round(sum(totals), 2)
            if abs(items_sum - round(schema.subtotal, 2)) > _TOLERANCE:
                report.add(
                    "line_item_sum", "warning",
                    f"Line items sum to {items_sum} but subtotal = {schema.subtotal}",
                )

    # Future date
    invoice_dt = _parse_date(schema.invoice_date)
    if invoice_dt and invoice_dt > now:
        report.add("future_date", "warning", f"Invoice date {schema.invoice_date} is in the future")

    # Stale invoice (>365 days old)
    if invoice_dt and (now - invoice_dt).days > 365:
        report.add(
            "stale_invoice", "info",
            f"Invoice date {schema.invoice_date} is more than 365 days old",
        )

    # Missing critical fields
    if not schema.vendor_name:
        report.add("missing_vendor", "warning", "vendor_name is missing")
    if not schema.invoice_number:
        report.add("missing_invoice_number", "info", "invoice_number is missing")
    if schema.total_amount is None:
        report.add("missing_total", "critical", "total_amount could not be extracted")

    return report
