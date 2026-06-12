import json
from pathlib import Path

from store import (discover_invoices, delete_invoice, save_extraction, load_extraction,
                    all_extractions_dataframe, schema_to_dfs, schema_from_dfs)
from models.invoice import InvoiceSchema, LineItem


def _make_pdf_store(base: Path, sha: str, pdf_name: str):
    vs = base / "vectorstore" / sha
    vs.mkdir(parents=True)
    (vs / "bm25.json").write_text(json.dumps({"texts": ["x"], "pages": [1]}))
    d = base / "data" / sha
    d.mkdir(parents=True)
    (d / pdf_name).write_bytes(b"%PDF-1.4 fake")


def test_discover_finds_pdf_invoices(tmp_path):
    _make_pdf_store(tmp_path, "abc12345", "acme_march.pdf")
    invoices = discover_invoices(tmp_path)
    assert "abc12345" in invoices
    assert invoices["abc12345"]["name"] == "acme_march.pdf"
    assert invoices["abc12345"]["type"] == "pdf"
    assert invoices["abc12345"]["sha_key"] == "abc12345"
    assert invoices["abc12345"]["schema_cache"] is None


def test_discover_skips_partial_vectorstore(tmp_path):
    (tmp_path / "vectorstore" / "deadbeef").mkdir(parents=True)  # no bm25.json
    assert discover_invoices(tmp_path) == {}


def test_discover_finds_images(tmp_path):
    img_dir = tmp_path / "data" / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "receipt.png").write_bytes(b"fake")
    invoices = discover_invoices(tmp_path)
    assert "img_receipt.png" in invoices
    assert invoices["img_receipt.png"]["type"] == "image"


def test_discover_empty_dir(tmp_path):
    assert discover_invoices(tmp_path) == {}


def test_delete_pdf_removes_all_files(tmp_path):
    _make_pdf_store(tmp_path, "abc12345", "acme.pdf")
    inv = discover_invoices(tmp_path)["abc12345"]
    delete_invoice(inv, tmp_path)
    assert not (tmp_path / "vectorstore" / "abc12345").exists()
    assert not (tmp_path / "data" / "abc12345").exists()


def test_delete_image_removes_file(tmp_path):
    img_dir = tmp_path / "data" / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "receipt.png").write_bytes(b"fake")
    inv = discover_invoices(tmp_path)["img_receipt.png"]
    delete_invoice(inv, tmp_path)
    assert not (img_dir / "receipt.png").exists()


def _schema():
    return InvoiceSchema(
        vendor_name="ACME", invoice_number="A-1", total_amount=110.0, currency="USD",
        line_items=[LineItem(description="Widget", quantity=2, unit_price=50.0, total=100.0)],
    )


def test_save_and_load_pdf_extraction(tmp_path):
    _make_pdf_store(tmp_path, "abc12345", "acme.pdf")
    inv = discover_invoices(tmp_path)["abc12345"]
    save_extraction(inv, _schema(), tmp_path)
    loaded = load_extraction(inv, tmp_path)
    assert loaded is not None
    assert loaded.vendor_name == "ACME"
    assert loaded.line_items[0].total == 100.0


def test_discover_rehydrates_schema_cache(tmp_path):
    _make_pdf_store(tmp_path, "abc12345", "acme.pdf")
    inv = discover_invoices(tmp_path)["abc12345"]
    save_extraction(inv, _schema(), tmp_path)
    rediscovered = discover_invoices(tmp_path)["abc12345"]
    assert rediscovered["schema_cache"] is not None
    assert rediscovered["schema_cache"].invoice_number == "A-1"


def test_save_and_load_image_extraction(tmp_path):
    img_dir = tmp_path / "data" / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "receipt.png").write_bytes(b"fake")
    inv = discover_invoices(tmp_path)["img_receipt.png"]
    save_extraction(inv, _schema(), tmp_path)
    assert load_extraction(inv, tmp_path).vendor_name == "ACME"


def test_delete_image_removes_extraction_sidecar(tmp_path):
    img_dir = tmp_path / "data" / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "receipt.png").write_bytes(b"fake")
    inv = discover_invoices(tmp_path)["img_receipt.png"]
    save_extraction(inv, _schema(), tmp_path)
    delete_invoice(inv, tmp_path)
    assert not (img_dir / "receipt.png.extraction.json").exists()


def test_all_extractions_dataframe_one_row_per_line_item(tmp_path):
    invoices = {
        "k1": {"name": "a.pdf", "type": "pdf", "sha_key": "k1", "schema_cache": _schema()},
        "k2": {"name": "b.pdf", "type": "pdf", "sha_key": "k2", "schema_cache": None},
        "k3": {"name": "c.pdf", "type": "pdf", "sha_key": "k3",
               "schema_cache": InvoiceSchema(vendor_name="NoItems", total_amount=5.0)},
    }
    df = all_extractions_dataframe(invoices)
    assert len(df) == 2  # one line-item row for k1, one headers-only row for k3; k2 skipped
    assert set(df.columns) >= {"invoice", "vendor_name", "invoice_number", "total_amount",
                               "currency", "item_description", "item_quantity",
                               "item_unit_price", "item_total"}
    assert df.iloc[0]["item_description"] == "Widget"


def test_schema_dfs_round_trip():
    original = InvoiceSchema(
        vendor_name="ACME", invoice_number="A-1", subtotal=100.0, tax=10.0,
        total_amount=110.0, currency="USD", po_number="PO-7",
        line_items=[LineItem(description="Widget", quantity=2.0, unit_price=50.0, total=100.0)],
    )
    header_df, items_df = schema_to_dfs(original)
    rebuilt = schema_from_dfs(header_df, items_df)
    assert rebuilt == original


def test_schema_from_dfs_coerces_numeric_strings():
    original = InvoiceSchema(total_amount=110.0)
    header_df, items_df = schema_to_dfs(original)
    header_df.loc[header_df["Field"] == "Total", "Value"] = "212.09"
    rebuilt = schema_from_dfs(header_df, items_df)
    assert rebuilt.total_amount == 212.09


def test_schema_from_dfs_blank_strings_become_none():
    original = InvoiceSchema(vendor_name="ACME")
    header_df, items_df = schema_to_dfs(original)
    header_df.loc[header_df["Field"] == "Vendor", "Value"] = ""
    rebuilt = schema_from_dfs(header_df, items_df)
    assert rebuilt.vendor_name is None
