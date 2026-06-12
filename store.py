import shutil
from pathlib import Path

import pandas as pd

from models.invoice import InvoiceSchema, LineItem

_HEADER_ROWS: list[tuple[str, str, bool]] = [
    ("Vendor", "vendor_name", False), ("Invoice #", "invoice_number", False),
    ("Date", "invoice_date", False), ("Due Date", "due_date", False),
    ("Subtotal", "subtotal", True), ("Tax", "tax", True), ("Total", "total_amount", True),
    ("Currency", "currency", False), ("PO #", "po_number", False),
    ("Payment Terms", "payment_terms", False), ("Vendor Tax ID", "vendor_tax_id", False),
    ("Vendor Address", "vendor_address", False), ("Bill To", "bill_to", False),
]


def schema_to_dfs(schema: InvoiceSchema):
    header = {
        "Field": [label for label, _, _ in _HEADER_ROWS],
        "Value": [getattr(schema, field) for _, field, _ in _HEADER_ROWS],
    }
    line_items = [
        {"Description": li.description, "Qty": li.quantity,
         "Unit Price": li.unit_price, "Total": li.total}
        for li in schema.line_items
    ]
    return pd.DataFrame(header), pd.DataFrame(line_items) if line_items else pd.DataFrame()


def _coerce(value, numeric: bool):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if numeric:
        return float(str(value).replace(",", ""))
    return str(value)


def schema_from_dfs(header_df: pd.DataFrame, items_df: pd.DataFrame) -> InvoiceSchema:
    values = dict(zip(header_df["Field"], header_df["Value"]))
    fields = {
        field: _coerce(values.get(label), numeric)
        for label, field, numeric in _HEADER_ROWS
    }
    items = []
    if not items_df.empty:
        for _, row in items_df.iterrows():
            desc = _coerce(row.get("Description"), False)
            if desc is None:
                continue
            items.append(LineItem(
                description=desc,
                quantity=_coerce(row.get("Qty"), True),
                unit_price=_coerce(row.get("Unit Price"), True),
                total=_coerce(row.get("Total"), True),
            ))
    return InvoiceSchema(**fields, line_items=items)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def discover_invoices(base_dir: Path) -> dict:
    """Rebuild the invoice registry from what exists on disk, rehydrating any saved extractions."""
    invoices: dict = {}
    vs_root = base_dir / "vectorstore"
    data_root = base_dir / "data"

    if vs_root.exists():
        for sha_dir in sorted(vs_root.iterdir()):
            if not (sha_dir / "bm25.json").exists():
                continue  # partial/corrupt leftovers are not valid invoices
            sha = sha_dir.name
            pdf_dir = data_root / sha
            pdfs = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
            name = pdfs[0].name if pdfs else f"{sha}.pdf"
            invoices[sha] = {"name": name, "type": "pdf", "sha_key": sha, "schema_cache": None}

    img_dir = data_root / "images"
    if img_dir.exists():
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() in _IMAGE_SUFFIXES:
                invoices[f"img_{img.name}"] = {
                    "name": img.name, "type": "image", "path": img, "schema_cache": None,
                }

    for inv in invoices.values():
        inv["schema_cache"] = load_extraction(inv, base_dir)
    return invoices


def delete_invoice(inv: dict, base_dir: Path) -> None:
    """Remove an invoice's files from disk (PDF copy + vectorstore, or image)."""
    if inv["type"] == "pdf":
        try:
            # chromadb keeps sqlite handles open per-path in-process; release them
            # or Windows file locks leave a half-deleted, corrupt vectorstore
            from chromadb.api.client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        shutil.rmtree(base_dir / "vectorstore" / inv["sha_key"], ignore_errors=True)
        shutil.rmtree(base_dir / "data" / inv["sha_key"], ignore_errors=True)
    else:
        img_path = Path(inv["path"])
        img_path.unlink(missing_ok=True)
        Path(str(img_path) + ".extraction.json").unlink(missing_ok=True)


def _extraction_path(inv: dict, base_dir: Path) -> Path:
    if inv["type"] == "pdf":
        return base_dir / "vectorstore" / inv["sha_key"] / "extraction.json"
    return Path(str(inv["path"]) + ".extraction.json")


def save_extraction(inv: dict, schema: InvoiceSchema, base_dir: Path) -> None:
    path = _extraction_path(inv, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(schema.model_dump_json(indent=2), encoding="utf8")


def load_extraction(inv: dict, base_dir: Path) -> InvoiceSchema | None:
    path = _extraction_path(inv, base_dir)
    if not path.exists():
        return None
    try:
        return InvoiceSchema.model_validate_json(path.read_text(encoding="utf8"))
    except Exception:
        return None  # stale/corrupt sidecar must not break discovery


def all_extractions_dataframe(invoices: dict) -> pd.DataFrame:
    header_fields = [
        "vendor_name", "invoice_number", "invoice_date", "due_date", "subtotal",
        "tax", "total_amount", "currency", "po_number", "payment_terms",
    ]
    rows = []
    for inv in invoices.values():
        schema = inv.get("schema_cache")
        if schema is None:
            continue
        base = {"invoice": inv["name"], **{f: getattr(schema, f) for f in header_fields}}
        if schema.line_items:
            for li in schema.line_items:
                rows.append({**base, "item_description": li.description,
                             "item_quantity": li.quantity, "item_unit_price": li.unit_price,
                             "item_total": li.total})
        else:
            rows.append({**base, "item_description": None, "item_quantity": None,
                         "item_unit_price": None, "item_total": None})
    return pd.DataFrame(rows)
