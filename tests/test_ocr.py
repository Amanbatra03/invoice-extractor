import pytest
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from ingest import _extract_page_texts


def _make_image_only_pdf(path: Path, text: str | None = None):
    img = Image.new("RGB", (1000, 700), "white")
    if text:
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(60)
        draw.text((80, 280), text, fill="black", font=font)
    img.save(path, "PDF")


def test_text_pdf_does_not_invoke_ocr(invoice_pdf, monkeypatch):
    import rag.ocr

    def _boom(*a, **k):
        raise AssertionError("OCR should not run for text PDFs")

    monkeypatch.setattr(rag.ocr, "ocr_pdf_pages", _boom)
    texts = _extract_page_texts(invoice_pdf)
    assert any(t.strip() for t in texts)


def test_scanned_pdf_falls_back_to_ocr(tmp_path, monkeypatch):
    import rag.ocr
    pdf = tmp_path / "scan.pdf"
    _make_image_only_pdf(pdf)

    monkeypatch.setattr(rag.ocr, "ocr_pdf_pages", lambda p: ["INVOICE TOTAL 212.09"])
    texts = _extract_page_texts(pdf)
    assert texts == ["INVOICE TOTAL 212.09"]


@pytest.mark.slow
def test_real_ocr_reads_rendered_text(tmp_path):
    from rag.ocr import ocr_pdf_pages
    pdf = tmp_path / "scan.pdf"
    _make_image_only_pdf(pdf, text="INVOICE 61356291")
    texts = ocr_pdf_pages(pdf)
    assert len(texts) == 1
    assert "61356291" in texts[0].replace(" ", "")
