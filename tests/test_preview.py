import pytest
from rag.preview import render_pdf_page


def test_render_pdf_page_returns_image(invoice_pdf):
    img = render_pdf_page(invoice_pdf)
    assert img.width > 100 and img.height > 100
