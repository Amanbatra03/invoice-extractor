import hashlib
import json
from pathlib import Path

import chromadb

from rag.utils import load_config


def _sha_key(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:8]


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

    # Copy PDF to data dir
    dest_dir = base_dir / "data" / sha_key
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (original_name or pdf_path.name)
    if not dest.exists():
        dest.write_bytes(pdf_path.read_bytes())

    # Heavy imports (torch, transformers) deferred so importing this module stays cheap
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader

    # Load PDF and split per page (page numbers are 1-based for display)
    reader = PdfReader(str(pdf_path))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP
    )
    texts: list[str] = []
    pages: list[int] = []
    for page_num, page in enumerate(reader.pages, start=1):
        for piece in splitter.split_text(page.extract_text() or ""):
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
