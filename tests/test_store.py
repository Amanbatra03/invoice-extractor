import json
from pathlib import Path

from store import discover_invoices, delete_invoice


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
