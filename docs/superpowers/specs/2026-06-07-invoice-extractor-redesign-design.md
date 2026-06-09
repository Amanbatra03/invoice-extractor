# Invoice Extractor — Unified Redesign Design Spec
**Date:** 2026-06-07  
**Status:** Approved

---

## Overview

Replace two disconnected Streamlit sub-apps with a single unified analyst-tool application. The PDF path gains a LangGraph-based agentic RAG loop with hybrid BM25 + dense retrieval (RRF fusion). New features: structured extraction via Pydantic schema, multi-invoice comparison, and a data-heavy analyst dashboard UI. All 9 security and correctness bugs from the code review are fixed.

---

## Architecture

Single entry point `app.py`. Two document input paths share one UI.

```
app.py (Streamlit)
  Sidebar: upload, invoice manager, config
  Main:    [Q&A] [Extract] [Compare] tabs
      │                        │
  PDF path                Image path
  rag/agent.py            vision/gemini.py
  (LangGraph loop)        (Gemini 1.5 Flash)
      │
  rag/hybrid_retriever.py
  BM25 (sparse) + ChromaDB (dense) → RRF fusion
      │
  rag/extractor.py        rag/comparator.py
  (Pydantic schema)       (multi-invoice diff)
```

**File structure:**
```
invoice-extractor/
├── app.py
├── config.yml
├── ingest.py
├── vision/
│   └── gemini.py
├── rag/
│   ├── agent.py
│   ├── hybrid_retriever.py
│   ├── extractor.py
│   ├── comparator.py
│   └── utils.py              # shared: JSON extraction, config loader
└── models/
    └── invoice.py
```

---

## Agentic RAG Loop

LangGraph graph with 5 nodes:

```
START → query_rewriter → hybrid_retrieve → relevance_grade
           ↑ (retry ≤2)        │
           └── all irrelevant ─┘
                                │ chunks relevant
                                ▼
                        generate_answer → self_critique → END
                               ↑ (retry ≤1)      │
                               └── not grounded ──┘
```

**Nodes:**
- `query_rewriter` — rewrites or decomposes complex multi-part questions into focused sub-queries
- `hybrid_retrieve` — BM25 + ChromaDB retrieval, RRF fusion, returns top-N chunks with scores
- `relevance_grade` — LLM scores each chunk relevant/irrelevant; filters before generation
- `generate_answer` — generates answer grounded strictly in relevant chunks
- `self_critique` — LLM checks answer is supported by retrieved context; loops if not

**RRF fusion:**
- Score: `1/(k + rank_bm25) + 1/(k + rank_dense)`, k=60
- BM25 weight: 0.4 (exact match — invoice numbers, amounts, dates)
- Dense weight: 0.6 (semantic — "freight cost" → "shipping charge")

---

## Hybrid Retrieval

**Ingest pipeline (ingest.py):**
1. Load PDFs with PyPDF, chunk with RecursiveCharacterTextSplitter
2. Build ChromaDB collection keyed by `sha256(filename)[:8]` — one collection per invoice
3. Build BM25 index (rank-bm25) over same chunks, persist as pickle alongside vectorstore

**Retrieval (rag/hybrid_retriever.py):**
- Query both indexes independently
- Merge ranked lists via RRF
- Return top `NUM_RESULTS` chunks with fused score, source page, and chunk text

**Config additions:**
```yaml
NUM_RESULTS: 4
BM25_WEIGHT: 0.4
DENSE_WEIGHT: 0.6
MAX_AGENT_ITERATIONS: 3
CHUNK_SIZE: 800
CHUNK_OVERLAP: 80
```

---

## Structured Extraction

**Pydantic schema (models/invoice.py):**
```python
class LineItem(BaseModel):
    description: str
    quantity: float | None
    unit_price: float | None
    total: float | None

class InvoiceSchema(BaseModel):
    vendor_name: str | None
    invoice_number: str | None
    invoice_date: str | None
    due_date: str | None
    subtotal: float | None
    tax: float | None
    total_amount: float | None
    currency: str | None
    line_items: list[LineItem]
```

**Extract tab flow:**
1. "Extract All Fields" button triggers `rag/extractor.py`
2. Extractor retrieves all chunks for the invoice, prompts LLM to fill schema as JSON
3. LLM output parsed into `InvoiceSchema` via `model_validate_json`
4. Displayed as two st.dataframe tables: header fields + line items
5. Export buttons: Download JSON, Download CSV

**Image path:** Gemini Vision receives the extraction prompt with schema — returns JSON directly.

---

## Multi-Invoice Comparison

**Management:**
- Each PDF upload stored in `data/<sha256[:8]>/filename.pdf` where sha256 is computed from file **contents** (not filename) for dedup
- Each gets its own ChromaDB collection: `invoice_<sha256[:8]>`
- BM25 index stored at `vectorstore/<sha256[:8]>/bm25.pkl`
- Sidebar lists all loaded invoices (PDF and image) with per-file delete and re-ingest buttons

**Compare tab flow:**
1. User selects 2+ **PDF** invoices via checkboxes in sidebar (image invoices use Gemini for Q&A and extraction but are excluded from RAG-based comparison)
2. `rag/comparator.py` runs structured extraction on each (result cached in `st.session_state`)
3. Renders side-by-side st.dataframe: rows = fields, columns = invoice names
4. Discrepancies highlighted: different vendor, total mismatch >5%, date gaps >30 days

---

## UI Layout

```
┌─ Sidebar ──────────────────┐  ┌─ Main ─────────────────────────────┐
│ Upload (PDF or Image)      │  │  [Q&A]  [Extract]  [Compare]       │
│                            │  │                                     │
│ Loaded Invoices            │  │  Q&A Tab:                          │
│  • INV-001.pdf  [✕]       │  │  Question input                    │
│  • INV-002.pdf  [✕]       │  │  Answer                            │
│  • invoice.jpg  [✕]       │  │  ▶ Agent reasoning trace (expand)  │
│                            │  │  Source chunks with RRF scores     │
│ Config                     │  │  and page references               │
│  NUM_RESULTS  [4    ]      │  │                                     │
│  MAX_ITERS    [3    ]      │  │  Extract Tab:                      │
│  Device  [cpu ▼]          │  │  Header fields table               │
└────────────────────────────┘  │  Line items table                  │
                                │  [Download JSON] [Download CSV]    │
                                │                                     │
                                │  Compare Tab:                      │
                                │  Side-by-side field diff table     │
                                │  Discrepancies highlighted red     │
                                └─────────────────────────────────────┘
```

---

## Error Handling

| Failure | Fix | User-facing message |
|---|---|---|
| Path traversal via filename | `Path(name).name` strips dirs | Silent (sanitized) |
| rmtree before success | Write to temp dir, rename on success | Old DB preserved on failure |
| Gemini safety block | `try/except` + `st.warning` | "Gemini blocked this request: {reason}" |
| Ollama not running | `try/except` + `st.error` | "Start Ollama: run `ollama serve` in a terminal" |
| Non-image file to Gemini | `PIL.Image.verify()` before API call | "File does not appear to be a valid image" |
| Config from wrong directory | `Path(__file__).parent / "config.yml"` everywhere | FileNotFoundError eliminated |
| Hardcoded vectorstore path | Read `VECTOR_DB` from config at startup | Config drift eliminated |
| JSON extraction bug | Shared `rag/utils.py` with `re.search(r'\{.*\}', s, re.DOTALL)` | Silent fix |
| NUM_RESULTS=1 | Raised to 4, exposed in sidebar | User can tune live |
| No MIME validation | `PIL.Image.verify()` | "File does not appear to be a valid image" |

---

## Dependencies (additions to requirements.txt)

```
langgraph>=0.2.0
rank-bm25>=0.2.2
```

Existing deps retained. `python-box`, `pyyaml`, `chromadb`, `sentence-transformers`, `langchain*`, `streamlit`, `pillow`, `google-generativeai`, `python-dotenv` all stay.

---

## Out of Scope

- Authentication / user accounts
- Cloud deployment / hosted vector DB
- Fine-tuned embeddings
- Evaluation harness / ground-truth testing
- Async processing / job queues
