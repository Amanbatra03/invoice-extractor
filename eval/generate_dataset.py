"""Generate deterministic synthetic invoices (PDF + ground-truth JSON)."""
import random
from pathlib import Path

from fpdf import FPDF

from models.invoice import InvoiceSchema, LineItem

_VENDORS = [
    ("Hartley Office Supply", "HOS-2291-T", "Net 30"),
    ("Brightline Logistics", "BLL-8830-X", "Net 15"),
    ("Cascade IT Services", "CIS-1104-R", "Due on receipt"),
    ("Meridian Catering Co", "MCC-5512-B", "Net 45"),
]
_ITEMS = [
    ("A4 paper ream 80gsm", 4.25), ("Wireless keyboard", 38.90),
    ("Server rack shelf", 112.50), ("Catering lunch tray", 14.75),
    ("Network patch cable 3m", 6.40), ("Monitor stand dual", 54.20),
    ("Coffee beans 1kg", 19.80), ("Whiteboard markers x10", 8.95),
]


def _make_truth(rng: random.Random, idx: int) -> InvoiceSchema:
    vendor, tax_id, terms = rng.choice(_VENDORS)
    items = []
    for _ in range(rng.randint(2, 4)):
        desc, price = rng.choice(_ITEMS)
        qty = rng.randint(1, 6)
        items.append(LineItem(description=desc, quantity=float(qty),
                              unit_price=price, total=round(qty * price, 2)))
    subtotal = round(sum(li.total for li in items), 2)
    tax = round(subtotal * 0.10, 2)
    return InvoiceSchema(
        vendor_name=vendor,
        invoice_number=f"INV-{2026}{idx:04d}",
        invoice_date=f"2026-{rng.randint(1, 6):02d}-{rng.randint(1, 28):02d}",
        due_date=None,
        subtotal=subtotal, tax=tax, total_amount=round(subtotal + tax, 2),
        currency="USD", po_number=f"PO-{rng.randint(1000, 9999)}",
        payment_terms=terms, vendor_tax_id=tax_id,
        line_items=items,
    )


def _render_pdf(truth: InvoiceSchema, path: Path) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "INVOICE", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Vendor: {truth.vendor_name}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Tax ID: {truth.vendor_tax_id}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Invoice Number: {truth.invoice_number}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Invoice Date: {truth.invoice_date}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"PO Number: {truth.po_number}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Payment Terms: {truth.payment_terms}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(90, 7, "Description"); pdf.cell(25, 7, "Qty"); pdf.cell(35, 7, "Unit Price")
    pdf.cell(30, 7, "Total", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for li in truth.line_items:
        pdf.cell(90, 7, li.description); pdf.cell(25, 7, f"{li.quantity:.0f}")
        pdf.cell(35, 7, f"{li.unit_price:.2f}")
        pdf.cell(30, 7, f"{li.total:.2f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.cell(0, 7, f"Subtotal: {truth.subtotal:.2f} USD", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Tax (10%): {truth.tax:.2f} USD", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Total Due: {truth.total_amount:.2f} USD", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def generate_dataset(out_dir: Path, n: int = 12, seed: int = 42) -> list[tuple[Path, Path]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pairs = []
    for i in range(1, n + 1):
        truth = _make_truth(rng, i)
        pdf_path = out_dir / f"inv_{i:02d}.pdf"
        truth_path = out_dir / f"inv_{i:02d}.json"
        _render_pdf(truth, pdf_path)
        truth_path.write_text(truth.model_dump_json(indent=2), encoding="utf8")
        pairs.append((pdf_path, truth_path))
    return pairs


if __name__ == "__main__":
    pairs = generate_dataset(Path(__file__).parent / "dataset")
    print(f"generated {len(pairs)} invoices in eval/dataset/")
