# Invoice Extractor — Unified Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified analyst-grade invoice extraction app from scratch in `C:\Users\amanb\invoice-extractor\`, replacing two disconnected Streamlit sub-apps with one polished tool.

**Architecture:** Single `app.py` entry point with sidebar invoice management and three tabs (Q&A, Extract, Compare). PDF invoices go through a 5-node LangGraph agentic RAG loop with hybrid BM25 + ChromaDB retrieval. Image invoices use Gemini 1.5 Flash directly. Structured Pydantic extraction and multi-invoice comparison are first-class features.

**Tech Stack:** Python 3.11+, Streamlit, LangGraph, LangChain Community, ChromaDB, rank-bm25, sentence-transformers (all-MiniLM-L6-v2), Ollama (local LLM), Google Gemini 1.5 Flash, Pydantic v2, PyPDF, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `requirements.txt` | All dependencies |
| `config.yml` | Runtime config: chunk size, model names, retrieval params |
| `.env.example` | Template for GOOGLE_API_KEY |
| `ingest.py` | Per-PDF ingestion: SHA256 keying, ChromaDB + BM25 creation |
| `models/__init__.py` | Package marker |
| `models/invoice.py` | Pydantic `InvoiceSchema` + `LineItem` |
| `rag/__init__.py` | Package marker |
| `rag/utils.py` | `load_config()`, `extract_json_from_text()` |
| `rag/hybrid_retriever.py` | `HybridRetriever` class + `rrf_score()` |
| `rag/agent.py` | LangGraph 5-node agentic RAG graph |
| `rag/extractor.py` | Structured extraction → `InvoiceSchema` |
| `rag/comparator.py` | Multi-invoice field comparison + discrepancy detection |
| `vision/__init__.py` | Package marker |
| `vision/gemini.py` | Gemini 1.5 Flash Q&A + extraction for image invoices |
| `app.py` | Unified Streamlit application |
| `tests/conftest.py` | `invoice_pdf` session fixture, slow marker |
| `tests/fixtures/invoice_1.pdf` | Test fixture PDF (copy from original project) |
| `tests/test_models.py` | Pydantic schema unit tests |
| `tests/test_utils.py` | Utils unit tests |
| `tests/test_ingest.py` | Ingest integration tests (marked slow) |
| `tests/test_hybrid_retriever.py` | RRF unit tests + retrieval integration (marked slow) |
| `tests/test_agent.py` | Agent flow tests with mock LLM |
| `tests/test_extractor.py` | Extractor tests with mock LLM |
| `tests/test_comparator.py` | Pure-logic comparator tests |
| `tests/test_vision.py` | Vision module tests with mock genai |
| `tests/test_app.py` | Streamlit smoke test |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `config.yml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `models/__init__.py`
- Create: `rag/__init__.py`
- Create: `vision/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/` (directory)

- [ ] **Step 1: Create requirements.txt**

```
langchain>=0.2.0
langchain-community>=0.2.0
langchain-core>=0.2.0
langchain-text-splitters>=0.2.0
langgraph>=0.2.0
chromadb>=0.5.0
sentence-transformers>=2.7.0
pypdf>=4.0.0
rank-bm25>=0.2.2
python-box>=7.0.0
pyyaml>=6.0
streamlit>=1.35.0
google-generativeai>=0.7.0
python-dotenv>=1.0.0
Pillow>=10.0.0
pydantic>=2.0.0
pandas>=2.0.0
pytest>=8.0.0
```

- [ ] **Step 2: Create config.yml**

```yaml
CHUNK_SIZE: 800
CHUNK_OVERLAP: 80
NUM_RESULTS: 4
BM25_WEIGHT: 0.4
DENSE_WEIGHT: 0.6
MAX_AGENT_ITERATIONS: 3
MAX_CRITIQUE_ITERATIONS: 1
DATA_PATH: "data"
VECTORSTORE_PATH: "vectorstore"
EMBEDDINGS: "sentence-transformers/all-MiniLM-L6-v2"
NORMALIZE_EMBEDDINGS: true
DEVICE: "cpu"
VECTOR_SPACE: "cosine"
LLM: "llama3.2:3b"
```

- [ ] **Step 3: Create .env.example**

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

- [ ] **Step 4: Create .gitignore**

```
__pycache__/
*.pyc
.env
data/
vectorstore/
*.pkl
.pytest_cache/
```

- [ ] **Step 5: Create empty package markers**

Create these four empty files:
- `models/__init__.py`
- `rag/__init__.py`
- `vision/__init__.py`
- `tests/__init__.py`

- [ ] **Step 6: Copy test fixture PDF**

Copy `C:\Users\amanb\OneDrive\Documents\generative ai\Invoice exxtractor\Code_Files-20241013T163136Z-001\Code_Files\Invoice-extraction-pdf-data\data\invoice_1.pdf` to `tests/fixtures/invoice_1.pdf`.

- [ ] **Step 7: Install dependencies**

```
pip install -r requirements.txt
```

Expected: all packages install without error. Note: first run downloads `all-MiniLM-L6-v2` (~22MB).

- [ ] **Step 8: Commit**

```bash
git add requirements.txt config.yml .env.example .gitignore models/ rag/ vision/ tests/
git commit -m "chore: scaffold invoice extractor project structure"
```

---

## Task 2: Pydantic Models

**Files:**
- Create: `models/invoice.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_models.py`:
```python
import pytest
from models.invoice import InvoiceSchema, LineItem


def test_invoice_schema_full():
    data = {
        "vendor_name": "ACME Corp",
        "invoice_number": "INV-001",
        "invoice_date": "2024-01-15",
        "due_date": "2024-02-15",
        "subtotal": 100.0,
        "tax": 10.0,
        "total_amount": 110.0,
        "currency": "USD",
        "line_items": [
            {"description": "Widget", "quantity": 2.0, "unit_price": 50.0, "total": 100.0}
        ],
    }
    schema = InvoiceSchema(**data)
    assert schema.vendor_name == "ACME Corp"
    assert schema.total_amount == 110.0
    assert len(schema.line_items) == 1
    assert schema.line_items[0].description == "Widget"


def test_invoice_schema_all_none():
    schema = InvoiceSchema()
    assert schema.vendor_name is None
    assert schema.total_amount is None
    assert schema.line_items == []


def test_line_item_description_only():
    item = LineItem(description="Service Fee")
    assert item.description == "Service Fee"
    assert item.quantity is None
    assert item.unit_price is None
    assert item.total is None


def test_invoice_schema_from_json():
    json_str = '{"vendor_name": "Corp A", "total_amount": 55.5, "line_items": []}'
    schema = InvoiceSchema.model_validate_json(json_str)
    assert schema.vendor_name == "Corp A"
    assert schema.total_amount == 55.5
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_models.py -v
```

Expected: `ImportError: cannot import name 'InvoiceSchema'`

- [ ] **Step 3: Implement models/invoice.py**

```python
from pydantic import BaseModel


class LineItem(BaseModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    total: float | None = None


class InvoiceSchema(BaseModel):
    vendor_name: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    total_amount: float | None = None
    currency: str | None = None
    line_items: list[LineItem] = []
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_models.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add models/invoice.py tests/test_models.py
git commit -m "feat: add Pydantic InvoiceSchema and LineItem models"
```

---

## Task 3: Shared Utilities

**Files:**
- Create: `rag/utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_utils.py`:
```python
import pytest
from pathlib import Path
from rag.utils import extract_json_from_text, load_config


def test_extract_json_simple():
    text = 'Result: {"key": "value", "num": 42} done.'
    result = extract_json_from_text(text)
    assert result == '{"key": "value", "num": 42}'


def test_extract_json_multiline():
    text = 'Answer:\n{\n  "vendor": "ACME",\n  "total": 100.0\n}\ndone.'
    result = extract_json_from_text(text)
    assert result is not None
    assert '"vendor": "ACME"' in result


def test_extract_json_no_json():
    result = extract_json_from_text("No JSON here at all.")
    assert result is None


def test_load_config_returns_box(tmp_path):
    cfg_file = tmp_path / "config.yml"
    cfg_file.write_text("CHUNK_SIZE: 800\nNUM_RESULTS: 4\n")
    cfg = load_config(cfg_file)
    assert cfg.CHUNK_SIZE == 800
    assert cfg.NUM_RESULTS == 4
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_utils.py -v
```

Expected: `ImportError: cannot import name 'extract_json_from_text'`

- [ ] **Step 3: Implement rag/utils.py**

```python
import re
from pathlib import Path
import box
import yaml


def load_config(path: Path | str | None = None) -> box.Box:
    if path is None:
        path = Path(__file__).parent.parent / "config.yml"
    with open(path, "r", encoding="utf8") as f:
        return box.Box(yaml.safe_load(f))


def extract_json_from_text(text: str) -> str | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_utils.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add rag/utils.py tests/test_utils.py
git commit -m "feat: add shared utils (load_config, extract_json_from_text)"
```

---

## Task 4: Ingest Pipeline

**Files:**
- Create: `ingest.py`
- Create: `tests/conftest.py`
- Create: `tests/test_ingest.py`

- [ ] **Step 1: Create tests/conftest.py**

```python
import pytest
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (skip with -m 'not slow')")


@pytest.fixture(scope="session")
def invoice_pdf():
    p = FIXTURES_DIR / "invoice_1.pdf"
    assert p.exists(), f"Test fixture not found: {p}\nCopy invoice_1.pdf to tests/fixtures/"
    return p
```

- [ ] **Step 2: Write the failing tests**

`tests/test_ingest.py`:
```python
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
```

- [ ] **Step 3: Run tests to confirm they fail**

```
pytest tests/test_ingest.py -v -m slow
```

Expected: `ImportError: cannot import name 'ingest_pdf'`

- [ ] **Step 4: Implement ingest.py**

```python
import hashlib
import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from rag.utils import load_config


def _sha_key(pdf_path: Path) -> str:
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:8]


def ingest_pdf(pdf_path: Path, base_dir: Path = Path("."), force: bool = False) -> str:
    cfg = load_config()
    sha_key = _sha_key(pdf_path)

    vectorstore_dir = base_dir / "vectorstore" / sha_key
    bm25_path = vectorstore_dir / "bm25.pkl"

    if bm25_path.exists() and not force:
        return sha_key

    dest_dir = base_dir / "data" / sha_key
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / pdf_path.name
    if not dest.exists():
        dest.write_bytes(pdf_path.read_bytes())

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]

    embeddings = HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDINGS,
        model_kwargs={"device": cfg.DEVICE},
        encode_kwargs={"normalize_embeddings": cfg.NORMALIZE_EMBEDDINGS},
    )

    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=f"invoice_{sha_key}",
        collection_metadata={"hnsw:space": cfg.VECTOR_SPACE},
        persist_directory=str(vectorstore_dir),
    )

    tokenized = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts, "chunks": chunks}, f)

    return sha_key
```

- [ ] **Step 5: Run tests to confirm they pass**

```
pytest tests/test_ingest.py -v -m slow
```

Expected: 5 passed. Note: first run downloads the embedding model and takes ~60s.

- [ ] **Step 6: Commit**

```bash
git add ingest.py tests/conftest.py tests/test_ingest.py
git commit -m "feat: add per-invoice ingest pipeline with SHA256 keying and BM25 index"
```

---

## Task 5: Hybrid Retriever

**Files:**
- Create: `rag/hybrid_retriever.py`
- Create: `tests/test_hybrid_retriever.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_hybrid_retriever.py`:
```python
import pickle
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag.hybrid_retriever import rrf_score, HybridRetriever
from ingest import ingest_pdf


def test_rrf_score_zero_ranks():
    score = rrf_score(rank_bm25=0, rank_dense=0, k=60)
    assert abs(score - 2 / 60) < 1e-9


def test_rrf_score_lower_rank_gives_higher_score():
    high = rrf_score(rank_bm25=0, rank_dense=0)
    low = rrf_score(rank_bm25=10, rank_dense=10)
    assert high > low


def test_rrf_score_custom_k():
    score = rrf_score(rank_bm25=0, rank_dense=0, k=10)
    assert abs(score - 2 / 10) < 1e-9


@pytest.mark.slow
def test_retrieve_returns_sorted_results(invoice_pdf, tmp_path):
    sha_key = ingest_pdf(invoice_pdf, base_dir=tmp_path)
    retriever = HybridRetriever(sha_key, base_dir=tmp_path)
    results = retriever.retrieve("What is the total amount due?")
    assert len(results) > 0
    assert all(k in results[0] for k in ("text", "page", "score"))
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.slow
def test_retrieve_returns_at_most_num_results(invoice_pdf, tmp_path):
    sha_key = ingest_pdf(invoice_pdf, base_dir=tmp_path)
    retriever = HybridRetriever(sha_key, base_dir=tmp_path)
    results = retriever.retrieve("invoice number")
    assert len(results) <= 4  # NUM_RESULTS in config
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_hybrid_retriever.py -v
```

Expected: `ImportError: cannot import name 'rrf_score'`

- [ ] **Step 3: Implement rag/hybrid_retriever.py**

```python
import pickle
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from rag.utils import load_config


def rrf_score(rank_bm25: int, rank_dense: int, k: int = 60) -> float:
    return 1 / (k + rank_bm25) + 1 / (k + rank_dense)


class HybridRetriever:
    def __init__(self, sha_key: str, base_dir: Path = Path(".")):
        cfg = load_config()
        self._num_results = cfg.NUM_RESULTS
        vectorstore_dir = base_dir / "vectorstore" / sha_key

        bm25_path = vectorstore_dir / "bm25.pkl"
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._texts = data["texts"]
        self._chunks = data["chunks"]

        embeddings = HuggingFaceEmbeddings(
            model_name=cfg.EMBEDDINGS,
            model_kwargs={"device": cfg.DEVICE},
            encode_kwargs={"normalize_embeddings": cfg.NORMALIZE_EMBEDDINGS},
        )
        self._vectorstore = Chroma(
            collection_name=f"invoice_{sha_key}",
            persist_directory=str(vectorstore_dir),
            collection_metadata={"hnsw:space": cfg.VECTOR_SPACE},
            embedding_function=embeddings,
        )

    def retrieve(self, query: str) -> list[dict]:
        n = min(self._num_results * 3, len(self._texts))

        tokenized_query = query.split()
        bm25_scores = self._bm25.get_scores(tokenized_query)
        sorted_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ranks = {idx: rank for rank, idx in enumerate(sorted_bm25_idx[:n])}

        dense_docs = self._vectorstore.similarity_search(query, k=n)
        dense_ranks: dict[int, int] = {}
        for rank, doc in enumerate(dense_docs):
            for i, text in enumerate(self._texts):
                if doc.page_content == text and i not in dense_ranks:
                    dense_ranks[i] = rank
                    break

        all_indices = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        fused = []
        for idx in all_indices:
            score = rrf_score(
                bm25_ranks.get(idx, n + 60),
                dense_ranks.get(idx, n + 60),
            )
            chunk = self._chunks[idx]
            fused.append({
                "text": self._texts[idx],
                "page": chunk.metadata.get("page", "?"),
                "score": score,
            })

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[: self._num_results]
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_hybrid_retriever.py -v
```

Fast tests (rrf_score): immediately pass. Slow tests: require fixture ingest (~60s first run).

```
pytest tests/test_hybrid_retriever.py -v -m slow
```

Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add rag/hybrid_retriever.py tests/test_hybrid_retriever.py
git commit -m "feat: add HybridRetriever with BM25 + ChromaDB RRF fusion"
```

---

## Task 6: LangGraph Agent

**Files:**
- Create: `rag/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_agent.py`:
```python
import pytest
from unittest.mock import MagicMock
from rag.agent import build_agent, AgentState


class _MockLLM:
    def __init__(self, responses: dict):
        self._responses = responses

    def invoke(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        for keyword, response in self._responses.items():
            if keyword in prompt_lower:
                return response
        return "default response"


def _make_mock_retriever(chunks):
    r = MagicMock()
    r.retrieve.return_value = chunks
    return r


def test_agent_happy_path():
    chunks = [{"text": "Total amount is $110.00", "page": 1, "score": 0.9}]
    # Keywords match actual prompt text: "rewrite", "relevant", "use the following", "supported"
    llm = _MockLLM({
        "rewrite": "What is the total amount?",
        "relevant": "yes",
        "use the following": "The total is $110.00",
        "supported": "yes",
    })
    retriever = _make_mock_retriever(chunks)
    agent = build_agent(retriever, llm=llm)

    result = agent.invoke({
        "query": "What is the total?",
        "rewritten_query": "",
        "chunks": [],
        "answer": "",
        "relevant": False,
        "grounded": False,
        "iterations": 0,
        "critique_iterations": 0,
    })

    assert "answer" in result
    assert result["answer"] != ""


def test_agent_retries_on_irrelevant_then_gives_up():
    llm = _MockLLM({
        "rewrite": "What is the invoice number?",
        "relevant": "no",
        "use the following": "I could not find that information.",
        "supported": "yes",
    })
    retriever = _make_mock_retriever([{"text": "unrelated text", "page": 1, "score": 0.1}])
    agent = build_agent(retriever, llm=llm)

    result = agent.invoke({
        "query": "What is the invoice number?",
        "rewritten_query": "",
        "chunks": [],
        "answer": "",
        "relevant": False,
        "grounded": False,
        "iterations": 0,
        "critique_iterations": 0,
    })

    assert result["iterations"] >= 1
    assert "answer" in result


def test_agent_self_critique_accepts_grounded_answer():
    chunks = [{"text": "Invoice date: 2024-01-15", "page": 1, "score": 0.95}]
    llm = _MockLLM({
        "rewrite": "What is the invoice date?",
        "relevant": "yes",
        "use the following": "The invoice date is 2024-01-15.",
        "supported": "yes",
    })
    retriever = _make_mock_retriever(chunks)
    agent = build_agent(retriever, llm=llm)

    result = agent.invoke({
        "query": "What is the invoice date?",
        "rewritten_query": "",
        "chunks": [],
        "answer": "",
        "relevant": False,
        "grounded": False,
        "iterations": 0,
        "critique_iterations": 0,
    })

    assert result["grounded"] is True
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_agent.py -v
```

Expected: `ImportError: cannot import name 'build_agent'`

- [ ] **Step 3: Implement rag/agent.py**

```python
from typing import TypedDict

from langgraph.graph import StateGraph, END

from rag.hybrid_retriever import HybridRetriever
from rag.utils import load_config


class AgentState(TypedDict):
    query: str
    rewritten_query: str
    chunks: list[dict]
    answer: str
    relevant: bool
    grounded: bool
    iterations: int
    critique_iterations: int


def build_agent(retriever: HybridRetriever, llm=None):
    cfg = load_config()

    if llm is None:
        from langchain_community.llms import Ollama
        llm = Ollama(model=cfg.LLM, temperature=0)

    def query_rewriter(state: AgentState) -> AgentState:
        prompt = (
            f"Rewrite this invoice question to be specific and extractable.\n"
            f"Original: {state['query']}\nRewritten:"
        )
        rewritten = llm.invoke(prompt).strip()
        return {
            **state,
            "rewritten_query": rewritten,
            "iterations": state.get("iterations", 0) + 1,
        }

    def hybrid_retrieve(state: AgentState) -> AgentState:
        chunks = retriever.retrieve(state["rewritten_query"])
        return {**state, "chunks": chunks}

    def relevance_grade(state: AgentState) -> AgentState:
        if not state["chunks"]:
            return {**state, "relevant": False}
        context = "\n".join(c["text"][:200] for c in state["chunks"])
        prompt = (
            f"Query: {state['rewritten_query']}\n"
            f"Context: {context}\n"
            f"Is the context relevant to answer the query? Reply ONLY 'yes' or 'no'."
        )
        verdict = llm.invoke(prompt).strip().lower()
        return {**state, "relevant": verdict.startswith("yes")}

    def route_from_grade(state: AgentState) -> str:
        if state["relevant"]:
            return "generate"
        if state.get("iterations", 0) >= cfg.MAX_AGENT_ITERATIONS:
            return "generate"
        return "retry"

    def generate_answer(state: AgentState) -> AgentState:
        context = "\n\n".join(c["text"] for c in state["chunks"])
        prompt = (
            f"Use the following invoice context to answer the question.\n"
            f"If the answer is not present, say 'I could not find that information in the invoice.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['query']}\nAnswer:"
        )
        answer = llm.invoke(prompt).strip()
        return {**state, "answer": answer}

    def self_critique(state: AgentState) -> AgentState:
        context = "\n\n".join(c["text"] for c in state["chunks"])
        prompt = (
            f"Context:\n{context}\n\n"
            f"Answer: {state['answer']}\n\n"
            f"Is this answer directly supported by the context? Reply ONLY 'yes' or 'no'."
        )
        verdict = llm.invoke(prompt).strip().lower()
        return {
            **state,
            "grounded": verdict.startswith("yes"),
            "critique_iterations": state.get("critique_iterations", 0) + 1,
        }

    def route_from_critique(state: AgentState) -> str:
        if state["grounded"]:
            return "end"
        if state.get("critique_iterations", 0) > cfg.MAX_CRITIQUE_ITERATIONS:
            return "end"
        return "retry"

    graph = StateGraph(AgentState)
    graph.add_node("query_rewriter", query_rewriter)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("relevance_grade", relevance_grade)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("self_critique", self_critique)

    graph.set_entry_point("query_rewriter")
    graph.add_edge("query_rewriter", "hybrid_retrieve")
    graph.add_edge("hybrid_retrieve", "relevance_grade")
    graph.add_conditional_edges(
        "relevance_grade",
        route_from_grade,
        {"generate": "generate_answer", "retry": "query_rewriter"},
    )
    graph.add_edge("generate_answer", "self_critique")
    graph.add_conditional_edges(
        "self_critique",
        route_from_critique,
        {"end": END, "retry": "generate_answer"},
    )

    return graph.compile()
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_agent.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add rag/agent.py tests/test_agent.py
git commit -m "feat: add LangGraph 5-node agentic RAG agent"
```

---

## Task 7: Structured Extractor

**Files:**
- Create: `rag/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_extractor.py`:
```python
import pytest
from unittest.mock import MagicMock
from rag.extractor import extract_invoice
from models.invoice import InvoiceSchema


def _make_retriever(text: str):
    r = MagicMock()
    r.retrieve.return_value = [{"text": text, "page": 1, "score": 0.9}]
    return r


def test_extract_returns_invoice_schema():
    retriever = _make_retriever("Total: $110.00, Vendor: ACME Corp, Invoice #INV-001")
    llm = MagicMock()
    llm.invoke.return_value = (
        '{"vendor_name": "ACME Corp", "invoice_number": "INV-001", '
        '"invoice_date": null, "due_date": null, "subtotal": null, '
        '"tax": null, "total_amount": 110.0, "currency": "USD", "line_items": []}'
    )
    result = extract_invoice(retriever, llm)
    assert isinstance(result, InvoiceSchema)
    assert result.vendor_name == "ACME Corp"
    assert result.total_amount == 110.0


def test_extract_returns_empty_schema_on_bad_json():
    retriever = _make_retriever("some invoice text")
    llm = MagicMock()
    llm.invoke.return_value = "I cannot extract the fields."
    result = extract_invoice(retriever, llm)
    assert isinstance(result, InvoiceSchema)
    assert result.vendor_name is None


def test_extract_returns_empty_schema_on_invalid_schema():
    retriever = _make_retriever("some invoice text")
    llm = MagicMock()
    llm.invoke.return_value = '{"completely": "wrong", "structure": 123}'
    result = extract_invoice(retriever, llm)
    assert isinstance(result, InvoiceSchema)
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_extractor.py -v
```

Expected: `ImportError: cannot import name 'extract_invoice'`

- [ ] **Step 3: Implement rag/extractor.py**

```python
from rag.hybrid_retriever import HybridRetriever
from rag.utils import extract_json_from_text
from models.invoice import InvoiceSchema

_EXTRACTION_PROMPT = """You are an invoice data extractor. Extract all fields from the invoice context below.
Return ONLY a valid JSON object with these exact keys (use null for missing values):
{{
  "vendor_name": string or null,
  "invoice_number": string or null,
  "invoice_date": string or null,
  "due_date": string or null,
  "subtotal": number or null,
  "tax": number or null,
  "total_amount": number or null,
  "currency": string or null,
  "line_items": [
    {{"description": string, "quantity": number or null, "unit_price": number or null, "total": number or null}}
  ]
}}

Invoice context:
{context}"""


def extract_invoice(
    retriever: HybridRetriever, llm, query: str = "extract all invoice fields"
) -> InvoiceSchema:
    chunks = retriever.retrieve(query)
    context = "\n\n".join(c["text"] for c in chunks)
    raw = llm.invoke(_EXTRACTION_PROMPT.format(context=context))
    json_str = extract_json_from_text(raw)
    if json_str is None:
        return InvoiceSchema()
    try:
        return InvoiceSchema.model_validate_json(json_str)
    except Exception:
        return InvoiceSchema()
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_extractor.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add rag/extractor.py tests/test_extractor.py
git commit -m "feat: add structured invoice extractor returning InvoiceSchema"
```

---

## Task 8: Multi-Invoice Comparator

**Files:**
- Create: `rag/comparator.py`
- Create: `tests/test_comparator.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_comparator.py`:
```python
import pytest
from models.invoice import InvoiceSchema
from rag.comparator import compare_invoices


def _schema(**kwargs) -> InvoiceSchema:
    return InvoiceSchema(**kwargs)


def test_compare_identical_invoices_no_discrepancies():
    s = _schema(vendor_name="ACME", total_amount=100.0, invoice_date="2024-01-15")
    result = compare_invoices([("inv1", s), ("inv2", s)])
    assert result["discrepancies"] == []


def test_compare_different_vendors_flagged():
    a = _schema(vendor_name="ACME Corp")
    b = _schema(vendor_name="Beta Ltd")
    result = compare_invoices([("a", a), ("b", b)])
    fields = [d["field"] for d in result["discrepancies"]]
    assert "vendor_name" in fields


def test_compare_total_mismatch_over_5pct_flagged():
    a = _schema(total_amount=100.0)
    b = _schema(total_amount=110.0)
    result = compare_invoices([("a", a), ("b", b)])
    fields = [d["field"] for d in result["discrepancies"]]
    assert "total_amount" in fields


def test_compare_total_mismatch_under_5pct_not_flagged():
    a = _schema(total_amount=100.0)
    b = _schema(total_amount=102.0)
    result = compare_invoices([("a", a), ("b", b)])
    fields = [d["field"] for d in result["discrepancies"]]
    assert "total_amount" not in fields


def test_compare_date_gap_over_30_days_flagged():
    a = _schema(invoice_date="2024-01-01")
    b = _schema(invoice_date="2024-02-15")
    result = compare_invoices([("a", a), ("b", b)])
    fields = [d["field"] for d in result["discrepancies"]]
    assert "invoice_date" in fields


def test_compare_date_gap_under_30_days_not_flagged():
    a = _schema(invoice_date="2024-01-01")
    b = _schema(invoice_date="2024-01-20")
    result = compare_invoices([("a", a), ("b", b)])
    fields = [d["field"] for d in result["discrepancies"]]
    assert "invoice_date" not in fields


def test_compare_returns_table_with_all_fields():
    a = _schema(vendor_name="ACME", total_amount=100.0)
    b = _schema(vendor_name="ACME", total_amount=100.0)
    result = compare_invoices([("a", a), ("b", b)])
    assert "vendor_name" in result["table"]
    assert result["table"]["vendor_name"] == {"a": "ACME", "b": "ACME"}


def test_compare_single_invoice_returns_empty():
    s = _schema(vendor_name="ACME")
    result = compare_invoices([("only", s)])
    assert result == {"table": {}, "discrepancies": []}
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_comparator.py -v
```

Expected: `ImportError: cannot import name 'compare_invoices'`

- [ ] **Step 3: Implement rag/comparator.py**

```python
from datetime import datetime
from models.invoice import InvoiceSchema

_FIELDS = [
    "vendor_name", "invoice_number", "invoice_date", "due_date",
    "subtotal", "tax", "total_amount", "currency",
]

_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y")


def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def compare_invoices(named_schemas: list[tuple[str, InvoiceSchema]]) -> dict:
    if len(named_schemas) < 2:
        return {"table": {}, "discrepancies": []}

    table = {
        field: {name: getattr(schema, field) for name, schema in named_schemas}
        for field in _FIELDS
    }

    discrepancies: list[dict] = []

    vendors = {v for v in table["vendor_name"].values() if v}
    if len(vendors) > 1:
        discrepancies.append({
            "field": "vendor_name",
            "detail": f"Different vendors: {', '.join(vendors)}",
        })

    totals = [(name, val) for name, val in table["total_amount"].items() if val is not None]
    if len(totals) >= 2:
        amounts = [v for _, v in totals]
        min_a, max_a = min(amounts), max(amounts)
        if min_a > 0 and (max_a - min_a) / min_a > 0.05:
            discrepancies.append({
                "field": "total_amount",
                "detail": f"Total mismatch >5%: {[f'{n}={v}' for n, v in totals]}",
            })

    parsed_dates = [
        (name, d)
        for name, val in table["invoice_date"].items()
        if (d := _parse_date(val)) is not None
    ]
    if len(parsed_dates) >= 2:
        date_values = [d for _, d in parsed_dates]
        gap = (max(date_values) - min(date_values)).days
        if gap > 30:
            discrepancies.append({
                "field": "invoice_date",
                "detail": f"Date gap of {gap} days between invoices",
            })

    return {"table": table, "discrepancies": discrepancies}
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_comparator.py -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add rag/comparator.py tests/test_comparator.py
git commit -m "feat: add multi-invoice comparator with discrepancy detection"
```

---

## Task 9: Vision Module (Gemini)

**Files:**
- Create: `vision/gemini.py`
- Create: `tests/test_vision.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_vision.py`:
```python
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from vision.gemini import ask_invoice, extract_invoice_gemini
from models.invoice import InvoiceSchema

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_mock_genai(text_response: str):
    mock_response = MagicMock()
    mock_response.text = text_response
    mock_response.prompt_feedback = None
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    return mock_model


def test_ask_invoice_returns_text(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"")

    with patch("vision.gemini._get_model") as mock_get, \
         patch("vision.gemini._validate_image") as mock_val:
        mock_get.return_value = _make_mock_genai("The total is $110.00")
        mock_val.return_value = MagicMock()

        result = ask_invoice(fake_image, "What is the total?")

    assert result == "The total is $110.00"


def test_ask_invoice_raises_on_gemini_block(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"")

    mock_feedback = MagicMock()
    mock_feedback.block_reason = "SAFETY"
    mock_response = MagicMock()
    mock_response.text = ""
    mock_response.prompt_feedback = mock_feedback
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    with patch("vision.gemini._get_model", return_value=mock_model), \
         patch("vision.gemini._validate_image") as mock_val:
        mock_val.return_value = MagicMock()
        with pytest.raises(RuntimeError, match="Gemini blocked"):
            ask_invoice(fake_image, "What is the total?")


def test_extract_invoice_gemini_returns_schema(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"")

    json_resp = (
        '{"vendor_name": "Test Corp", "invoice_number": "T-001", '
        '"invoice_date": null, "due_date": null, "subtotal": null, '
        '"tax": null, "total_amount": 99.0, "currency": "USD", "line_items": []}'
    )

    with patch("vision.gemini._get_model") as mock_get, \
         patch("vision.gemini._validate_image") as mock_val:
        mock_get.return_value = _make_mock_genai(json_resp)
        mock_val.return_value = MagicMock()

        result = extract_invoice_gemini(fake_image)

    assert isinstance(result, InvoiceSchema)
    assert result.vendor_name == "Test Corp"
    assert result.total_amount == 99.0


def test_extract_invoice_gemini_returns_empty_on_bad_response(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"")

    with patch("vision.gemini._get_model") as mock_get, \
         patch("vision.gemini._validate_image") as mock_val:
        mock_get.return_value = _make_mock_genai("Sorry, cannot extract.")
        mock_val.return_value = MagicMock()

        result = extract_invoice_gemini(fake_image)

    assert isinstance(result, InvoiceSchema)
    assert result.vendor_name is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_vision.py -v
```

Expected: `ImportError: cannot import name 'ask_invoice'`

- [ ] **Step 3: Implement vision/gemini.py**

```python
import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

from models.invoice import InvoiceSchema
from rag.utils import extract_json_from_text

load_dotenv()

_SYSTEM_PROMPT = (
    "You are an expert invoice analyst. "
    "Answer questions using only information visible in the invoice image. "
    "If the requested information is not present, say 'Not found in invoice.'"
)

_EXTRACTION_PROMPT = """Extract all available invoice fields from this image.
Return ONLY a valid JSON object with these exact keys (use null for missing values):
{
  "vendor_name": string or null,
  "invoice_number": string or null,
  "invoice_date": string or null,
  "due_date": string or null,
  "subtotal": number or null,
  "tax": number or null,
  "total_amount": number or null,
  "currency": string or null,
  "line_items": [
    {"description": string, "quantity": number or null, "unit_price": number or null, "total": number or null}
  ]
}"""


def _get_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not set. Add it to your .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-flash")


def _validate_image(image_path: Path) -> Image.Image:
    try:
        img = Image.open(image_path)
        img.verify()
        return Image.open(image_path)
    except Exception as exc:
        raise ValueError(f"File does not appear to be a valid image: {exc}") from exc


def ask_invoice(image_path: Path, question: str) -> str:
    model = _get_model()
    img = _validate_image(image_path)
    response = model.generate_content([_SYSTEM_PROMPT, img, question])
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        raise RuntimeError(f"Gemini blocked this request: {response.prompt_feedback.block_reason}")
    return response.text


def extract_invoice_gemini(image_path: Path) -> InvoiceSchema:
    try:
        model = _get_model()
        img = _validate_image(image_path)
        response = model.generate_content([_EXTRACTION_PROMPT, img])
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return InvoiceSchema()
        json_str = extract_json_from_text(response.text)
        if json_str:
            return InvoiceSchema.model_validate_json(json_str)
    except Exception:
        pass
    return InvoiceSchema()
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_vision.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add vision/gemini.py tests/test_vision.py
git commit -m "feat: add Gemini vision module for image invoice Q&A and extraction"
```

---

## Task 10: Unified Streamlit App

**Files:**
- Create: `app.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write the smoke test first**

`tests/test_app.py`:
```python
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception


def test_app_has_sidebar():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception


def test_app_has_three_tabs():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert len(at.tabs) == 3
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_app.py -v
```

Expected: `FileNotFoundError` or `ModuleNotFoundError` — app.py doesn't exist yet.

- [ ] **Step 3: Implement app.py**

```python
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ingest import ingest_pdf
from models.invoice import InvoiceSchema
from rag.agent import AgentState, build_agent
from rag.comparator import compare_invoices
from rag.extractor import extract_invoice
from rag.hybrid_retriever import HybridRetriever
from rag.utils import load_config
from vision.gemini import ask_invoice, extract_invoice_gemini

load_dotenv()
cfg = load_config()

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="Invoice Analyst", page_icon="🧾", layout="wide")

# Session state
if "invoices" not in st.session_state:
    st.session_state["invoices"] = {}


def _get_ollama_llm():
    from langchain_community.llms import Ollama
    return Ollama(model=cfg.LLM, temperature=0)


def _schema_to_dfs(schema: InvoiceSchema):
    header = {
        "Field": ["Vendor", "Invoice #", "Date", "Due Date", "Subtotal", "Tax", "Total", "Currency"],
        "Value": [
            schema.vendor_name, schema.invoice_number, schema.invoice_date,
            schema.due_date, schema.subtotal, schema.tax, schema.total_amount, schema.currency,
        ],
    }
    line_items = [
        {"Description": li.description, "Qty": li.quantity, "Unit Price": li.unit_price, "Total": li.total}
        for li in schema.line_items
    ]
    return pd.DataFrame(header), pd.DataFrame(line_items) if line_items else pd.DataFrame()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Invoice")
    uploaded = st.file_uploader("PDF or Image", type=["pdf", "jpg", "jpeg", "png"])

    if uploaded and st.button("Add Invoice", type="primary"):
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        if suffix == ".pdf":
            with st.spinner(f"Ingesting {uploaded.name}…"):
                try:
                    sha_key = ingest_pdf(tmp_path, base_dir=BASE_DIR)
                    st.session_state["invoices"][sha_key] = {
                        "name": uploaded.name,
                        "type": "pdf",
                        "sha_key": sha_key,
                        "schema_cache": None,
                    }
                    st.success(f"Ingested: {uploaded.name}")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
        else:
            img_dest = BASE_DIR / "data" / "images"
            img_dest.mkdir(parents=True, exist_ok=True)
            safe_name = Path(uploaded.name).name  # strip any directory components
            dest = img_dest / safe_name
            dest.write_bytes(uploaded.getvalue())
            img_key = f"img_{safe_name}"
            st.session_state["invoices"][img_key] = {
                "name": safe_name,
                "type": "image",
                "path": dest,
                "schema_cache": None,
            }
            st.success(f"Loaded: {uploaded.name}")

        tmp_path.unlink(missing_ok=True)

    st.divider()
    st.subheader("Loaded Invoices")
    to_delete = []
    for key, inv in st.session_state["invoices"].items():
        col1, col2 = st.columns([4, 1])
        col1.markdown(f"**{inv['name']}** `{inv['type']}`")
        if col2.button("✕", key=f"del_{key}"):
            to_delete.append(key)
    for k in to_delete:
        del st.session_state["invoices"][k]
        st.rerun()

    st.divider()
    st.subheader("Config")
    cfg.NUM_RESULTS = st.number_input("NUM_RESULTS", min_value=1, max_value=10, value=cfg.NUM_RESULTS)
    cfg.MAX_AGENT_ITERATIONS = st.number_input("MAX_ITERS", min_value=1, max_value=5, value=cfg.MAX_AGENT_ITERATIONS)
    cfg.DEVICE = st.selectbox("Device", ["cpu", "cuda"], index=0)

# ── Invoice selector ──────────────────────────────────────────────────────────
invoices = st.session_state["invoices"]
if not invoices:
    st.info("Upload an invoice in the sidebar to get started.")
    st.stop()

invoice_names = {k: v["name"] for k, v in invoices.items()}
selected_key = st.selectbox("Active invoice", list(invoice_names.keys()), format_func=lambda k: invoice_names[k])
selected = invoices[selected_key]

# ── Tabs ──────────────────────────────────────────────────────────────────────
qa_tab, extract_tab, compare_tab = st.tabs(["Q&A", "Extract", "Compare"])

# ── Q&A Tab ───────────────────────────────────────────────────────────────────
with qa_tab:
    question = st.text_input("Ask a question about this invoice", placeholder="e.g. What is the invoice total?")
    ask_btn = st.button("Ask", type="primary", key="ask_btn")

    if ask_btn and question.strip():
        if selected["type"] == "pdf":
            try:
                retriever = HybridRetriever(selected["sha_key"], base_dir=BASE_DIR)
                llm = _get_ollama_llm()
                agent = build_agent(retriever, llm=llm)
                trace_steps = []
                with st.spinner("Running agentic RAG…"):
                    for event in agent.stream({
                        "query": question,
                        "rewritten_query": "",
                        "chunks": [],
                        "answer": "",
                        "relevant": False,
                        "grounded": False,
                        "iterations": 0,
                        "critique_iterations": 0,
                    }):
                        trace_steps.append(event)
                final = trace_steps[-1] if trace_steps else {}
                state = list(final.values())[0] if final else {}
                st.subheader("Answer")
                st.success(state.get("answer", "No answer generated."))
                with st.expander("Agent reasoning trace"):
                    for step in trace_steps:
                        for node, updates in step.items():
                            st.markdown(f"**Node: `{node}`**")
                            st.json({k: str(v)[:300] for k, v in updates.items()})
                chunks = state.get("chunks", [])
                if chunks:
                    with st.expander("Source chunks"):
                        for i, c in enumerate(chunks, 1):
                            st.markdown(f"**Chunk {i}** — page `{c['page']}`, score `{c['score']:.4f}`")
                            st.text(c["text"][:400])
            except Exception as e:
                st.error(f"Error: {e}\n\nMake sure Ollama is running: `ollama serve`")
        else:
            try:
                with st.spinner("Asking Gemini…"):
                    answer = ask_invoice(selected["path"], question)
                st.subheader("Answer")
                st.success(answer)
            except EnvironmentError as e:
                st.error(str(e))
            except RuntimeError as e:
                st.warning(str(e))

# ── Extract Tab ───────────────────────────────────────────────────────────────
with extract_tab:
    if st.button("Extract All Fields", type="primary", key="extract_btn"):
        if selected["type"] == "pdf":
            try:
                retriever = HybridRetriever(selected["sha_key"], base_dir=BASE_DIR)
                llm = _get_ollama_llm()
                with st.spinner("Extracting structured fields…"):
                    schema = extract_invoice(retriever, llm)
                invoices[selected_key]["schema_cache"] = schema
            except Exception as e:
                st.error(f"Extraction failed: {e}\n\nMake sure Ollama is running: `ollama serve`")
        else:
            try:
                with st.spinner("Extracting via Gemini…"):
                    schema = extract_invoice_gemini(selected["path"])
                invoices[selected_key]["schema_cache"] = schema
            except EnvironmentError as e:
                st.error(str(e))

    cached = invoices[selected_key].get("schema_cache")
    if cached:
        header_df, items_df = _schema_to_dfs(cached)
        st.subheader("Header Fields")
        st.dataframe(header_df, use_container_width=True, hide_index=True)
        if not items_df.empty:
            st.subheader("Line Items")
            st.dataframe(items_df, use_container_width=True, hide_index=True)
        col1, col2 = st.columns(2)
        col1.download_button(
            "Download JSON",
            data=cached.model_dump_json(indent=2),
            file_name=f"{selected['name']}_extracted.json",
            mime="application/json",
        )
        col2.download_button(
            "Download CSV",
            data=header_df.to_csv(index=False),
            file_name=f"{selected['name']}_extracted.csv",
            mime="text/csv",
        )

# ── Compare Tab ───────────────────────────────────────────────────────────────
with compare_tab:
    pdf_invoices = {k: v for k, v in invoices.items() if v["type"] == "pdf"}
    if len(pdf_invoices) < 2:
        st.info("Load at least 2 PDF invoices to compare.")
    else:
        selected_for_compare = []
        st.markdown("Select invoices to compare:")
        for key, inv in pdf_invoices.items():
            if st.checkbox(inv["name"], key=f"cmp_{key}"):
                selected_for_compare.append(key)

        if len(selected_for_compare) >= 2 and st.button("Compare Selected", type="primary"):
            named_schemas: list[tuple[str, InvoiceSchema]] = []
            for key in selected_for_compare:
                inv = invoices[key]
                schema = inv.get("schema_cache")
                if schema is None:
                    try:
                        retriever = HybridRetriever(inv["sha_key"], base_dir=BASE_DIR)
                        llm = _get_ollama_llm()
                        with st.spinner(f"Extracting {inv['name']}…"):
                            schema = extract_invoice(retriever, llm)
                        invoices[key]["schema_cache"] = schema
                    except Exception as e:
                        st.error(f"Failed to extract {inv['name']}: {e}")
                        schema = InvoiceSchema()
                named_schemas.append((inv["name"], schema))

            result = compare_invoices(named_schemas)
            table = result["table"]
            discrepancies = result["discrepancies"]

            if table:
                st.subheader("Side-by-Side Comparison")
                rows = []
                for field, values in table.items():
                    row = {"Field": field, **values}
                    rows.append(row)
                compare_df = pd.DataFrame(rows).set_index("Field")

                def highlight_discrepancies(df):
                    disc_fields = {d["field"] for d in discrepancies}
                    styles = pd.DataFrame("", index=df.index, columns=df.columns)
                    for field in disc_fields:
                        if field in styles.index:
                            styles.loc[field] = "background-color: #ffcccc"
                    return styles

                st.dataframe(compare_df.style.apply(highlight_discrepancies, axis=None), use_container_width=True)

            if discrepancies:
                st.subheader("Discrepancies")
                for d in discrepancies:
                    st.warning(f"**{d['field']}**: {d['detail']}")
            else:
                st.success("No discrepancies found.")
```

- [ ] **Step 4: Run smoke tests**

```
pytest tests/test_app.py -v
```

Expected: 3 passed

- [ ] **Step 5: Run all fast tests to verify nothing broke**

```
pytest tests/ -v -m "not slow"
```

Expected: all pass (models, utils, agent, extractor, comparator, vision, app)

- [ ] **Step 6: Manual smoke test — start the app**

```
streamlit run app.py
```

Expected: app opens at `http://localhost:8501`. Verify:
- Sidebar shows upload widget and empty invoice list
- Three tabs visible: Q&A, Extract, Compare
- Uploading a PDF triggers ingestion spinner
- Uploaded invoice appears in sidebar list with delete button
- Config sliders visible in sidebar

- [ ] **Step 7: Manual end-to-end test with a real invoice**

1. Upload `tests/fixtures/invoice_1.pdf`
2. Switch to **Extract** tab → click "Extract All Fields" (requires `ollama serve` running)
3. Confirm header fields table and line items table render
4. Click "Download JSON" — verify JSON file downloads
5. Switch to **Q&A** tab → ask "What is the total amount?" — verify answer appears with agent trace
6. Load a second PDF → switch to **Compare** tab → select both → click "Compare Selected"
7. Verify side-by-side table renders, discrepancies highlighted red

- [ ] **Step 8: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add unified Streamlit app with Q&A, Extract, and Compare tabs"
```

---

## Final: Run Full Test Suite

- [ ] **Run all non-slow tests**

```
pytest tests/ -v -m "not slow"
```

Expected output (representative):
```
tests/test_models.py::test_invoice_schema_full PASSED
tests/test_models.py::test_invoice_schema_all_none PASSED
tests/test_models.py::test_line_item_description_only PASSED
tests/test_models.py::test_invoice_schema_from_json PASSED
tests/test_utils.py::test_extract_json_simple PASSED
tests/test_utils.py::test_extract_json_multiline PASSED
tests/test_utils.py::test_extract_json_no_json PASSED
tests/test_utils.py::test_load_config_returns_box PASSED
tests/test_hybrid_retriever.py::test_rrf_score_zero_ranks PASSED
tests/test_hybrid_retriever.py::test_rrf_score_lower_rank_gives_higher_score PASSED
tests/test_hybrid_retriever.py::test_rrf_score_custom_k PASSED
tests/test_agent.py::test_agent_happy_path PASSED
tests/test_agent.py::test_agent_retries_on_irrelevant_then_gives_up PASSED
tests/test_agent.py::test_agent_self_critique_accepts_grounded_answer PASSED
tests/test_extractor.py::test_extract_returns_invoice_schema PASSED
tests/test_extractor.py::test_extract_returns_empty_schema_on_bad_json PASSED
tests/test_extractor.py::test_extract_returns_empty_schema_on_invalid_schema PASSED
tests/test_comparator.py::... (8 tests) PASSED
tests/test_vision.py::... (4 tests) PASSED
tests/test_app.py::... (3 tests) PASSED
```

- [ ] **Run slow integration tests (requires embedding model + fixture PDF)**

```
pytest tests/ -v -m slow
```

Expected: 7 slow tests pass

- [ ] **Final commit**

```bash
git add .
git commit -m "chore: all tests passing — invoice extractor redesign complete"
```
