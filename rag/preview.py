from pathlib import Path


def render_pdf_page(pdf_path: Path, page_index: int = 0, scale: float = 1.5):
    """First-page preview as a PIL image (pypdfium2 — already a dependency)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_index]
        return page.render(scale=scale).to_pil()
    finally:
        pdf.close()
