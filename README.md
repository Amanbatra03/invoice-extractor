# Invoice Extractor

A unified analyst-grade invoice extraction tool with a dual-mode pipeline: a **fully offline RAG path** for PDFs (LangGraph agentic loop + hybrid retrieval) and a **Gemini Vision path** for images. Single Streamlit app with Q&A, structured extraction, and multi-invoice comparison.

---

## Features

- **Conversational Cross-Invoice Chat (API v2)** — persisted multi-turn conversations over all ingested invoices; a routed LangGraph agent condenses follow-up questions against history, answers aggregate questions ("which invoice has the highest total?") from structured extractions, cites file name + page for document details, and sends image invoices to Gemini vision
- **Liquid Glass UI** — the API-backed Streamlit frontend ships a glassmorphic theme: blurred translucent panels, animated depth backdrop, sprung message entrances, and a typing indicator (motion-safe via `prefers-reduced-motion`)
- **Agentic RAG for PDFs** — 5-node LangGraph graph: query rewriting → hybrid retrieval → relevance grading → answer generation → self-critique
- **Hybrid Retrieval** — BM25 (sparse) + ChromaDB (dense) fused via Reciprocal Rank Fusion (RRF)
- **Whole-Document Schema-Constrained Extraction** — full document fed to the LLM with JSON-mode output locked to the `InvoiceSchema`; no chunk-hunting
- **Arithmetic Validation** — post-extraction checks: line-item qty × unit price, sum of items vs subtotal, subtotal + tax vs total (2% tolerance)
- **OCR Fallback** — scanned/image-only PDFs are automatically rendered and OCR'd via pypdfium2 + RapidOCR when pypdf finds no text
- **Multi-Invoice Comparison** — side-by-side field diff with automatic discrepancy detection (vendor mismatch, total >5%, date gap >30 days)
- **Structured Extraction** — Pydantic v2 `InvoiceSchema` extracts vendor, invoice #, dates, totals, line items; exportable as JSON or CSV
- **Extraction Persistence** — extractions saved as sidecar JSON files; rehydrated on startup, no re-extraction needed
- **Gemini Vision** — Google Gemini 2.0 Flash for image invoices (JPG/PNG)
- **Fully Offline PDF Path** — Ollama local LLM + local embeddings, no API key required for PDF mode
- **Per-Invoice Dedup** — SHA-256 content hashing, re-upload skips re-ingestion
- **Password Gate** — optional `APP_PASSWORD` env var gates the Streamlit UI with a constant-time password check
- **PDF Preview** — first-page thumbnail rendered in the sidebar alongside each invoice

---

## Architecture

```
app.py (Streamlit) + auth.py (password gate)
  Sidebar: upload · invoice manager (store.py) · config
  ┌──────────────┬──────────────┬──────────────┐
  │   Q&A tab    │ Extract tab  │ Compare tab  │
  └──────┬───────┴──────┬───────┴──────┬───────┘
         │              │              │
    PDF path       PDF path       PDF path
    rag/agent.py   rag/extractor  rag/comparator
    (LangGraph)        │               │
         │         rag/validator.py    │
         │         (arithmetic checks) │
    rag/hybrid_retriever.py      models/invoice.py (Pydantic)
    BM25 + ChromaDB → RRF fusion
         │
    ingest.py (SHA256, ChromaDB + BM25)
    rag/ocr.py (pypdfium2 + RapidOCR fallback)
    rag/llm.py (Ollama factory, schema-constrained)
    rag/preview.py (first-page thumbnail)

    Image path → vision/gemini.py (Gemini 2.0 Flash)
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

### Run with Docker

```bash
docker compose up -d --build
docker compose exec ollama ollama pull llama3.2:3b   # first time only
```

Open `http://localhost:8501`. Set `GOOGLE_API_KEY` / `APP_PASSWORD` in your shell
or a `.env` file next to `docker-compose.yml` to pass them in.

---

## Usage

### PDF Invoices (offline)
1. Upload a PDF in the sidebar → click **Add Invoice**
2. **Q&A tab** — ask natural language questions; the agent retrieves relevant chunks, generates a grounded answer, and shows its reasoning trace
3. **Extract tab** — click **Extract All Fields** to populate a structured table; arithmetic warnings surface inline; download as JSON or CSV
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
| `NUM_CTX` | `8192` | Ollama context window (tokens) |
| `EMBEDDINGS` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM` | `llama3.2:3b` | Ollama model name |
| `DEVICE` | `cpu` | `cpu` or `cuda` |

---

## Project Structure

```
invoice-extractor/
├── app.py                  # Unified Streamlit application
├── auth.py                 # Optional password gate (APP_PASSWORD)
├── ingest.py               # PDF ingestion (SHA256, ChromaDB + BM25)
├── store.py                # Invoice persistence: discover, save, load, delete
├── config.yml              # Runtime configuration
├── models/
│   └── invoice.py          # Pydantic InvoiceSchema + LineItem
├── rag/
│   ├── agent.py            # LangGraph 5-node agentic RAG
│   ├── hybrid_retriever.py # BM25 + ChromaDB RRF fusion
│   ├── extractor.py        # Whole-document schema-constrained extraction
│   ├── validator.py        # Arithmetic validation checks
│   ├── comparator.py       # Multi-invoice comparison
│   ├── llm.py              # Ollama LLM factory (JSON-mode, configurable context)
│   ├── ocr.py              # OCR fallback via pypdfium2 + RapidOCR
│   ├── preview.py          # First-page PDF thumbnail
│   └── utils.py            # Shared: load_config, extract_json_from_text
├── vision/
│   └── gemini.py           # Gemini 2.0 Flash for image invoices
├── eval/
│   ├── generate_dataset.py # Synthetic invoice dataset generator
│   ├── run_eval.py         # Field-level accuracy evaluation runner
│   └── scoring.py          # Per-field scoring logic
└── tests/                  # 60+ tests (TDD, pytest)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Agent orchestration | LangGraph |
| LLM (local) | Ollama (`llama3.2:3b`) |
| LLM (vision) | Google Gemini 2.0 Flash |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | ChromaDB (PersistentClient) |
| Sparse retrieval | rank-bm25 (BM25Okapi) |
| PDF loading | pypdf |
| OCR | pypdfium2 + RapidOCR |
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

## Running the Evaluation Suite

```bash
# Requires Ollama running with the configured model
python -m eval.run_eval          # defaults to 12 synthetic invoices
python -m eval.run_eval --n 25   # larger run
```

Results are written to `eval/results.md` with per-field accuracy and an overall score.
