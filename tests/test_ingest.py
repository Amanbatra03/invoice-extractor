import json
import pytest
from pathlib import Path
from ingest import ingest_pdf


@pytest.mark.slow
def test_ingest_returns_8char_hex_key(invoice_pdf, tmp_path):
    sha_key = ingest_pdf(invoice_pdf, base_dir=tmp_path)
    assert len(sha_key) == 8
    assert all(c in "0123456789abcdef" for c in sha_key)


@pytest.mark.slow
def test_ingest_creates_data_dir(invoice_pdf, tmp_path):
    sha_key = ingest_pdf(invoice_pdf, base_dir=tmp_path)
    assert (tmp_path / "data" / sha_key).is_dir()
    assert (tmp_path / "data" / sha_key / invoice_pdf.name).exists()


@pytest.mark.slow
def test_ingest_creates_chromadb(invoice_pdf, tmp_path):
    sha_key = ingest_pdf(invoice_pdf, base_dir=tmp_path)
    assert (tmp_path / "vectorstore" / sha_key / "chroma.sqlite3").exists()


@pytest.mark.slow
def test_ingest_creates_bm25_index(invoice_pdf, tmp_path):
    sha_key = ingest_pdf(invoice_pdf, base_dir=tmp_path)
    index_path = tmp_path / "vectorstore" / sha_key / "bm25.json"
    assert index_path.exists()
    data = json.loads(index_path.read_text(encoding="utf8"))
    assert "texts" in data
    assert "pages" in data
    assert len(data["texts"]) > 0
    assert len(data["pages"]) == len(data["texts"])


@pytest.mark.slow
def test_ingest_dedup_same_content(invoice_pdf, tmp_path):
    key1 = ingest_pdf(invoice_pdf, base_dir=tmp_path)
    key2 = ingest_pdf(invoice_pdf, base_dir=tmp_path)
    assert key1 == key2
