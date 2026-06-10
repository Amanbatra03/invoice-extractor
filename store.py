import shutil
from pathlib import Path

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def discover_invoices(base_dir: Path) -> dict:
    """Rebuild the invoice registry from what exists on disk."""
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
        Path(inv["path"]).unlink(missing_ok=True)
