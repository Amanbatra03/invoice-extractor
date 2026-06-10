from pathlib import Path


def ocr_pdf_pages(pdf_path: Path, scale: float = 2.0) -> list[str]:
    """Render each PDF page to an image and OCR it. Heavy imports deferred."""
    import numpy as np
    import pypdfium2 as pdfium
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    texts: list[str] = []
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for page in pdf:
            pil_image = page.render(scale=scale).to_pil()
            result, _ = engine(np.array(pil_image))
            texts.append("\n".join(item[1] for item in result) if result else "")
    finally:
        pdf.close()
    return texts
