import pickle
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
    bm25_path = tmp_path / "vectorstore" / sha_key / "bm25.pkl"
    assert bm25_path.exists()
    with open(bm25_path, "rb") as f:
        data = pickle.load(f)
    assert "bm25" in data
    assert "texts" in data
    assert len(data["texts"]) > 0


@pytest.mark.slow
def test_ingest_dedup_same_content(invoice_pdf, tmp_path):
    key1 = ingest_pdf(invoice_pdf, base_dir=tmp_path)
    key2 = ingest_pdf(invoice_pdf, base_dir=tmp_path)
    assert key1 == key2
