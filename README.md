# Invoice Extractor

A unified analyst-grade invoice extraction tool with a dual-mode pipeline: a **fully offline RAG path** for PDFs (LangGraph agentic loop + hybrid retrieval) and a **Gemini Vision path** for images. Single Streamlit app with Q&A, structured extraction, and multi-invoice comparison.

---

## Features

- **Agentic RAG for PDFs** — 5-node LangGraph graph: query rewriting → hybrid retrieval → relevance grading → answer generation → self-critique
- **Hybrid Retrieval** — BM25 (sparse) + ChromaDB (dense) fused via Reciprocal Rank Fusion (RRF)
- **Structured Extraction** — Pydantic v2 schema extracts vendor, invoice #, dates, totals, line items; exportable as JSON or CSV
- **Multi-Invoice Comparison** — side-by-side field diff with automatic discrepancy detection (vendor mismatch, total >5%, date gap >30 days)
- **Gemini Vision** — Google Gemini 1.5 Flash for image invoices (JPG/PNG)
- **Fully Offline PDF Path** — Ollama local LLM + local embeddings, no API key required for PDF mode
- **Per-Invoice Dedup** — SHA-256 content hashing, re-upload skips re-ingestion

---

## Architecture

```
app.py (Streamlit)
  Sidebar: upload · invoice manager · config
  ┌──────────────┬──────────────┬──────────────┐
  │   Q&A tab    │ Extract tab  │ Compare tab  │
  └──────┬───────┴──────┬───────┴──────┬───────┘
         │              │              │
    PDF path       PDF path       PDF path
    rag/agent.py   rag/extractor  rag/comparator
    (LangGraph)        │               │
         │         models/invoice.py (Pydantic)
    rag/hybrid_retriever.py
    BM25 + ChromaDB → RRF fusion
         │
    ingest.py (SHA256 keying, per-invoice vectorstore)

    Image path → vision/gemini.py (Gemini 1.5 Flash)
```

### LangGraph Agent Flow

```
START → query_rewriter → hybrid_retrieve → relevance_grade
             ↑ retry ≤3         │
             └── irrelevant ────┘
                                │ relevant
                                ▼
                        generate_answer → self_critique → END
                               ↑ retry ≤1      │
                               └── not grounded ┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) (for PDF Q&A and extraction)
- Google Gemini API key (for image invoices only)

### Install

```bash
git clone https://github.com/Amanbatra03/invoice-extractor.git
cd invoice-extractor
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY (only needed for image invoices)
```

### Pull the local LLM

```bash
ollama pull llama3.2:3b
```

### Run

```bash
# Terminal 1 — start local LLM server
ollama serve

# Terminal 2 — start the app
streamlit run app.py
```

Open `http://localhost:8501`.

---

## Usage

### PDF Invoices (offline)
1. Upload a PDF in the sidebar → click **Add Invoice**
2. **Q&A tab** — ask natural language questions; the agent retrieves relevant chunks, generates a grounded answer, and shows its reasoning trace
3. **Extract tab** — click **Extract All Fields** to populate a structured table; download as JSON or CSV
4. **Compare tab** — load 2+ PDFs, check the ones to compare, click **Compare Selected**

### Image Invoices (Gemini)
1. Upload a JPG/PNG in the sidebar
2. **Q&A tab** — ask questions answered by Gemini Vision
3. **Extract tab** — Gemini extracts all structured fields

---

## Configuration

Edit `config.yml` to tune retrieval and generation:

| Key | Default | Description |
|-----|---------|-------------|
| `CHUNK_SIZE` | `800` | PDF chunk size (tokens) |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `NUM_RESULTS` | `4` | Chunks retrieved per query |
| `MAX_AGENT_ITERATIONS` | `3` | Max query-rewrite retries |
| `MAX_CRITIQUE_ITERATIONS` | `1` | Max self-critique retries |
| `EMBEDDINGS` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM` | `llama3.2:3b` | Ollama model name |
| `DEVICE` | `cpu` | `cpu` or `cuda` |

---

## Project Structure

```
invoice-extractor/
├── app.py                  # Unified Streamlit application
├── ingest.py               # PDF ingestion (SHA256 keying, ChromaDB + BM25)
├── config.yml              # Runtime configuration
├── models/
│   └── invoice.py          # Pydantic InvoiceSchema + LineItem
├── rag/
│   ├── agent.py            # LangGraph 5-node agentic RAG
│   ├── hybrid_retriever.py # BM25 + ChromaDB RRF fusion
│   ├── extractor.py        # Structured field extraction
│   ├── comparator.py       # Multi-invoice comparison
│   └── utils.py            # Shared: load_config, extract_json_from_text
├── vision/
│   └── gemini.py           # Gemini 1.5 Flash for image invoices
└── tests/                  # 31 tests (TDD, pytest)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Agent orchestration | LangGraph |
| LLM (local) | Ollama (`llama3.2:3b`) |
| LLM (vision) | Google Gemini 1.5 Flash |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | ChromaDB (PersistentClient) |
| Sparse retrieval | rank-bm25 (BM25Okapi) |
| PDF loading | PyPDF + LangChain |
| Schema validation | Pydantic v2 |
| Testing | pytest |

---

## Running Tests

```bash
# Fast tests only (no model downloads required)
pytest tests/ -m "not slow"

# Full suite including integration tests (downloads ~22MB embedding model on first run)
pytest tests/ -v
```
