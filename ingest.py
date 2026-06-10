import hashlib
import json
import shutil
from pathlib import Path

import chromadb

from rag.utils import load_config


def _sha_key(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:8]


_MIN_TEXT_CHARS = 32  # below this across all pages, treat the PDF as scanned


def _extract_page_texts(pdf_path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    page_texts = [(page.extract_text() or "") for page in reader.pages]
    if sum(len(t.strip()) for t in page_texts) < _MIN_TEXT_CHARS:
        import rag.ocr
        page_texts = rag.ocr.ocr_pdf_pages(pdf_path)
    return page_texts


def ingest_pdf(
    pdf_path: Path,
    base_dir: Path = Path("."),
    force: bool = False,
    original_name: str | None = None,
) -> str:
    cfg = load_config()
    sha_key = _sha_key(pdf_path)

    vectorstore_dir = base_dir / "vectorstore" / sha_key
    index_path = vectorstore_dir / "bm25.json"

    if index_path.exists() and not force:
        return sha_key

    # bm25.json is written last, so its absence means any existing vectorstore
    # dir is a partial/corrupt leftover from an interrupted ingestion — start fresh
    if vectorstore_dir.exists():
        shutil.rmtree(vectorstore_dir, ignore_errors=True)

    # Copy PDF to data dir
    dest_dir = base_dir / "data" / sha_key
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (original_name or pdf_path.name)
    if not dest.exists():
        dest.write_bytes(pdf_path.read_bytes())

    # Heavy imports (torch, transformers) deferred so importing this module stays cheap
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Load PDF (with OCR fallback for scans) and split per page (1-based pages)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP
    )
    texts: list[str] = []
    pages: list[int] = []
    for page_num, page_text in enumerate(_extract_page_texts(pdf_path), start=1):
        for piece in splitter.split_text(page_text):
            texts.append(piece)
            pages.append(page_num)

    # Build embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDINGS,
        model_kwargs={"device": cfg.DEVICE},
        encode_kwargs={"normalize_embeddings": cfg.NORMALIZE_EMBEDDINGS},
    )

    # Persist to ChromaDB using PersistentClient (chromadb >= 0.4)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(vectorstore_dir))
    collection = client.get_or_create_collection(
        name=f"invoice_{sha_key}",
        metadata={"hnsw:space": cfg.VECTOR_SPACE},
    )
    embeds = embeddings.embed_documents(texts)
    metadatas = [{"page": p} for p in pages]
    ids = [f"chunk_{i}" for i in range(len(texts))]
    collection.add(
        documents=texts,
        embeddings=embeds,
        metadatas=metadatas,
        ids=ids,
    )

    # Persist the BM25 corpus as JSON (rebuilt on load — avoids unpickling untrusted data)
    payload = {"texts": texts, "pages": pages}
    index_path.write_text(json.dumps(payload), encoding="utf8")

    return sha_key
