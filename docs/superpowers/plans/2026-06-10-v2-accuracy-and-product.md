# V2: Accuracy & Product Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the nine improvements converged on by the AI-architect / business-user / business-analyst reviews: loud extraction failures, whole-document schema-constrained extraction, retrieval+performance fixes, an extended currency-aware schema, persistent extractions with real exports, a synthetic eval harness with published accuracy numbers, document preview with inline correction, batch upload, and an analytics dashboard.

**Architecture:** `app.py` stays a thin UI layer; all new logic lands in focused modules (`store.py` grows persistence/export/df-mapping, `rag/extractor.py` reworked, `rag/preview.py` and `eval/` new). Extraction stops using retrieval (whole document fits in context) and uses schema-constrained decoding on both the Ollama and Gemini paths. Failures raise `ExtractionError` and are surfaced, never masked as empty results. TDD throughout, one commit per task minimum.

**Tech Stack:** existing stack + `fpdf2` (synthetic eval PDFs). No new services.

**Environment notes (this machine):**
- Windows 11, anaconda Python 3.11.9; run everything from `C:\Users\amanb\invoice-extractor`
- Native-wheel pins that must not be disturbed: torch 2.7.1+cpu, chromadb 0.6.3, onnxruntime 1.19.2, numpy 1.26.4, opencv-python 4.10.0.84. After ANY pip install, check `pip check`-style complaints mentioning numpy/onnxruntime.
- Ollama is running locally with `llama3.2:3b` pulled (needed for Task 6's eval run).
- Fast tests: `python -m pytest -m "not slow" -q`. Full: `python -m pytest -q` (~80s).
- Commits go directly to `master` and push to origin (session convention); end commit messages with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

**Task order matters:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9. Task 6 (eval) must run after 2/4 (extraction + schema final). Tasks 7 and 9 depend on Task 5's persistence.

---

## File Structure

```
rag/extractor.py        # MODIFY — whole-document context, ExtractionError, schema-constrained llm
rag/llm.py              # MODIFY — format_schema + num_ctx params
rag/hybrid_retriever.py # MODIFY — all_chunks(), normalized tokenizer, injectable embeddings
rag/agent.py            # MODIFY — feedback-injected rewrite retries
rag/ocr.py              # MODIFY — cached RapidOCR engine singleton
rag/preview.py          # NEW — render_pdf_page via pypdfium2
rag/comparator.py       # MODIFY — currency-aware totals, currency mismatch check, new fields
rag/validator.py        # MODIFY — has_amounts()
models/invoice.py       # MODIFY — po_number, payment_terms, vendor_tax_id, vendor_address, bill_to
vision/gemini.py        # MODIFY — response_schema structured output, raise on failure
store.py                # MODIFY — save/load_extraction, schema_to_dfs/schema_from_dfs, all_extractions_dataframe
app.py                  # MODIFY — error surfacing, cached embeddings, preview+editor, batch upload, Dashboard tab
config.yml              # MODIFY — drop dead BM25_WEIGHT/DENSE_WEIGHT, add NUM_CTX
eval/generate_dataset.py  # NEW — 12 synthetic labeled invoices (fpdf2)
eval/scoring.py           # NEW — field_match / score_invoice
eval/run_eval.py          # NEW — accuracy table → eval/results.md
tests/: test_extractor.py, test_llm.py, test_hybrid_retriever.py, test_agent.py,
        test_models.py, test_comparator.py, test_validator.py, test_vision.py,
        test_store.py, test_preview.py (NEW), test_eval.py (NEW), test_app.py
```

---

### Task 1: Extraction failure must be loud

Today bad LLM JSON silently returns `InvoiceSchema()`; the UI shows "—" plus **"All arithmetic checks pass."**, and Compare treats the empty schema as data ("No discrepancies found"). A finance tool must never present failure as success.

**Files:**
- Modify: `rag/extractor.py`, `vision/gemini.py`, `rag/validator.py`, `app.py`
- Test: `tests/test_extractor.py`, `tests/test_vision.py`, `tests/test_validator.py`

- [ ] **Step 1: Rewrite the extractor failure tests (replace the two "returns empty schema" tests)**

In `tests/test_extractor.py`, replace `test_extract_returns_empty_schema_on_bad_json` and `test_extract_returns_empty_schema_on_invalid_schema` with:

```python
import pytest
from rag.extractor import extract_invoice, ExtractionError


def test_extract_raises_on_bad_json():
    retriever = _make_retriever("some invoice text")
    llm = MagicMock()
    llm.invoke.return_value = "I cannot extract the fields."
    with pytest.raises(ExtractionError):
        extract_invoice(retriever, llm)


def test_extract_raises_on_invalid_schema():
    retriever = _make_retriever("some invoice text")
    llm = MagicMock()
    llm.invoke.return_value = '{"vendor_name": {"nested": "wrong type"}}'
    with pytest.raises(ExtractionError):
        extract_invoice(retriever, llm)
```

Keep the existing imports at the top of the file and merge (`pytest` may already be imported).

- [ ] **Step 2: Add validator has_amounts tests (append to tests/test_validator.py)**

```python
from rag.validator import has_amounts


def test_has_amounts_true_with_total():
    assert has_amounts(InvoiceSchema(total_amount=10.0)) is True


def test_has_amounts_true_with_line_item_total():
    assert has_amounts(InvoiceSchema(line_items=[LineItem(description="A", total=5.0)])) is True


def test_has_amounts_false_when_empty():
    assert has_amounts(InvoiceSchema()) is False
```

- [ ] **Step 3: Rewrite the Gemini failure test (replace test_extract_invoice_gemini_returns_empty_on_bad_response in tests/test_vision.py)**

```python
def test_extract_invoice_gemini_raises_on_bad_response(tmp_path):
    from rag.extractor import ExtractionError
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"")

    with patch("vision.gemini._get_client") as mock_get, \
         patch("vision.gemini._validate_image") as mock_val:
        mock_get.return_value = _make_mock_client("Sorry, cannot extract.")
        mock_val.return_value = MagicMock()
        with pytest.raises(ExtractionError):
            extract_invoice_gemini(fake_image)
```

- [ ] **Step 4: Run to verify failures**

Run: `python -m pytest tests/test_extractor.py tests/test_validator.py tests/test_vision.py -q`
Expected: FAIL — `ImportError: cannot import name 'ExtractionError'` / `cannot import name 'has_amounts'`

- [ ] **Step 5: Implement ExtractionError in rag/extractor.py**

Replace the body of `extract_invoice` (keep `_EXTRACTION_PROMPT` as is for now — Task 2/4 change it):

```python
class ExtractionError(RuntimeError):
    """The LLM did not produce a parseable, schema-valid extraction."""


def extract_invoice(
    retriever: HybridRetriever, llm, query: str = "extract all invoice fields"
) -> InvoiceSchema:
    chunks = retriever.retrieve(query)
    context = "\n\n".join(c["text"] for c in chunks)
    raw = llm.invoke(_EXTRACTION_PROMPT.format(context=context))
    json_str = extract_json_from_text(raw)
    if json_str is None:
        raise ExtractionError("Model response contained no JSON object.")
    try:
        return InvoiceSchema.model_validate_json(json_str)
    except Exception as exc:
        raise ExtractionError(f"Model JSON did not match the invoice schema: {exc}") from exc
```

- [ ] **Step 6: Implement has_amounts in rag/validator.py (append)**

```python
def has_amounts(schema: InvoiceSchema) -> bool:
    """True if the extraction contains at least one monetary value."""
    if any(getattr(schema, f) is not None for f in ("subtotal", "tax", "total_amount")):
        return True
    return any(li.total is not None or li.unit_price is not None for li in schema.line_items)
```

- [ ] **Step 7: Make vision/gemini.py raise instead of swallow**

Replace `extract_invoice_gemini`:

```python
from rag.extractor import ExtractionError   # add to imports


def extract_invoice_gemini(image_path: Path) -> InvoiceSchema:
    # Config/input errors (missing key, bad image) propagate as-is;
    # model-output problems raise ExtractionError so the UI can say "retry".
    client = _get_client()
    img = _validate_image(image_path)
    response = client.models.generate_content(
        model=_MODEL, contents=[_EXTRACTION_PROMPT, img]
    )
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        raise RuntimeError(f"Gemini blocked this request: {response.prompt_feedback.block_reason}")
    json_str = extract_json_from_text(response.text)
    if json_str is None:
        raise ExtractionError("Gemini response contained no JSON object.")
    try:
        return InvoiceSchema.model_validate_json(json_str)
    except Exception as exc:
        raise ExtractionError(f"Gemini JSON did not match the invoice schema: {exc}") from exc
```

(The `test_extract_invoice_gemini_raises_without_api_key` test still passes: `_get_client` raises before anything else.)

- [ ] **Step 8: Surface failures in app.py**

Extract tab — wrap both paths:

```python
        if st.button("Extract All Fields", type="primary", key="extract_btn"):
            from rag.extractor import ExtractionError
            try:
                if selected_ext["type"] == "pdf":
                    retriever = HybridRetriever(selected_ext["sha_key"], base_dir=BASE_DIR)
                    llm = _get_ollama_llm()
                    with st.spinner("Extracting structured fields…"):
                        schema = extract_invoice(retriever, llm)
                else:
                    with st.spinner("Extracting via Gemini…"):
                        schema = extract_invoice_gemini(selected_ext["path"])
                invoices[selected_key_ext]["schema_cache"] = schema
            except ExtractionError as e:
                st.error(f"Extraction failed — the model did not return usable data. Try again. ({e})")
            except (EnvironmentError, ValueError) as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Extraction failed: {e}\n\nMake sure Ollama is running: `ollama serve`")
```

Validation caption — replace the `checks` block:

```python
            from rag.validator import has_amounts
            checks = validate_invoice(cached)
            if checks:
                st.subheader("Validation checks")
                for w in checks:
                    st.warning(w)
            elif has_amounts(cached):
                st.caption("All arithmetic checks pass.")
            else:
                st.warning("No monetary fields were extracted — treat this result as incomplete.")
```

Compare tab — failed extraction must be excluded, not blanked. Replace the inner extraction loop:

```python
            from rag.extractor import ExtractionError
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
                    except (ExtractionError, Exception) as e:
                        st.error(f"{inv['name']} excluded from comparison — extraction failed: {e}")
                        continue
                named_schemas.append((inv["name"], schema))

            if len(named_schemas) < 2:
                st.warning("Need at least 2 successful extractions to compare.")
                st.stop()
```

- [ ] **Step 9: Run the fast suite**

Run: `python -m pytest -m "not slow" -q`
Expected: all pass

- [ ] **Step 10: Commit**

```bash
git add rag/extractor.py rag/validator.py vision/gemini.py app.py tests/test_extractor.py tests/test_validator.py tests/test_vision.py
git commit -m "fix: extraction failures raise ExtractionError and surface in UI"
```

---

### Task 2: Whole-document, schema-constrained extraction

Extraction currently retrieves top-4 chunks for a static query — line-item tables that don't match get dropped. The full corpus is already on disk and fits in context. Also: free-form "return ONLY JSON" prompting on a 3B model is the main parse-failure source; Ollama and Gemini both support schema-constrained decoding.

**Files:**
- Modify: `rag/hybrid_retriever.py` (add `all_chunks()`), `rag/extractor.py`, `rag/llm.py`, `vision/gemini.py`, `app.py`
- Test: `tests/test_hybrid_retriever.py`, `tests/test_extractor.py`, `tests/test_llm.py`, `tests/test_vision.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_hybrid_retriever.py`:

```python
@pytest.mark.slow
def test_all_chunks_returns_full_corpus_in_order(invoice_pdf, tmp_path):
    sha_key = ingest_pdf(invoice_pdf, base_dir=tmp_path)
    retriever = HybridRetriever(sha_key, base_dir=tmp_path)
    chunks = retriever.all_chunks()
    assert len(chunks) >= 1
    assert all(set(c) >= {"text", "page"} for c in chunks)
    full_text = " ".join(c["text"] for c in chunks)
    assert "212,09" in full_text or "212.09" in full_text  # the gross total survives
```

In `tests/test_extractor.py`, update `_make_retriever` and add a test:

```python
def _make_retriever(text: str):
    r = MagicMock()
    r.all_chunks.return_value = [{"text": text, "page": 1}]
    r.retrieve.return_value = [{"text": text, "page": 1, "score": 0.9}]
    return r


def test_extract_uses_whole_document_not_retrieval():
    retriever = _make_retriever("Total: $110.00")
    llm = MagicMock()
    llm.invoke.return_value = '{"vendor_name": "X", "line_items": []}'
    extract_invoice(retriever, llm)
    retriever.all_chunks.assert_called_once()
    retriever.retrieve.assert_not_called()
```

Append to `tests/test_llm.py`:

```python
def test_format_schema_passed_through(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    llm = get_ollama_llm("llama3.2:3b", format_schema=schema)
    assert llm.format == schema


def test_num_ctx_default():
    llm = get_ollama_llm("llama3.2:3b")
    assert llm.num_ctx == 8192
```

In `tests/test_vision.py`, append:

```python
def test_extract_invoice_gemini_uses_response_schema(tmp_path):
    fake_image = tmp_path / "test.jpg"
    fake_image.write_bytes(b"")
    mock_client = _make_mock_client('{"vendor_name": "Test Corp", "line_items": []}')

    with patch("vision.gemini._get_client", return_value=mock_client), \
         patch("vision.gemini._validate_image", return_value=MagicMock()):
        extract_invoice_gemini(fake_image)

    kwargs = mock_client.models.generate_content.call_args.kwargs
    assert kwargs.get("config", {}).get("response_mime_type") == "application/json"
```

- [ ] **Step 2: Run to verify failures**

Run: `python -m pytest tests/test_extractor.py tests/test_llm.py tests/test_vision.py -m "not slow" -q`
Expected: FAIL — `all_chunks` not called / unexpected `format_schema` kwarg / missing config kwarg

- [ ] **Step 3: Implement all_chunks() in rag/hybrid_retriever.py (append method to the class)**

```python
    def all_chunks(self) -> list[dict]:
        """The full corpus in page order — for whole-document extraction."""
        return [
            {"text": text, "page": page}
            for text, page in zip(self._texts, self._pages)
        ]
```

- [ ] **Step 4: Extend rag/llm.py**

```python
import os


def get_ollama_llm(model: str, temperature: float = 0, format_schema: dict | None = None,
                   num_ctx: int = 8192):
    from langchain_ollama import OllamaLLM

    kwargs = dict(
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,   # Ollama's default 2048 silently truncates long invoices
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    if format_schema is not None:
        kwargs["format"] = format_schema
    return OllamaLLM(**kwargs)
```

Fallback note for the executor: if the installed `langchain-ollama` rejects a dict for `format` (pydantic ValidationError at construction), change the line to `kwargs["format"] = "json"` and adjust `test_format_schema_passed_through` to assert `llm.format == "json"` — constrained-to-JSON is the floor, schema-constrained is the goal.

- [ ] **Step 5: Rework rag/extractor.py to whole-document context**

Replace `extract_invoice` (keep `ExtractionError` from Task 1):

```python
def extract_invoice(retriever: HybridRetriever, llm) -> InvoiceSchema:
    chunks = retriever.all_chunks()
    context = "\n\n".join(c["text"] for c in chunks)
    raw = llm.invoke(_EXTRACTION_PROMPT.format(context=context))
    json_str = extract_json_from_text(raw)
    if json_str is None:
        raise ExtractionError("Model response contained no JSON object.")
    try:
        return InvoiceSchema.model_validate_json(json_str)
    except Exception as exc:
        raise ExtractionError(f"Model JSON did not match the invoice schema: {exc}") from exc
```

(The `query` parameter is dropped; nothing else calls it with a query.)

- [ ] **Step 6: Schema-constrain the Gemini path in vision/gemini.py**

Replace the `generate_content` call inside `extract_invoice_gemini`:

```python
    response = client.models.generate_content(
        model=_MODEL,
        contents=[_EXTRACTION_PROMPT, img],
        config={
            "response_mime_type": "application/json",
            "response_schema": InvoiceSchema,
        },
    )
```

(Keep the existing `extract_json_from_text` + `model_validate_json` parsing — it works for both structured and plain responses, and the mocks in tests return plain text.)

- [ ] **Step 7: Use the constrained llm in app.py Extract/Compare paths**

In both places that extract from PDFs (`Extract` tab and `Compare` loop), replace `llm = _get_ollama_llm()` with:

```python
                    llm = get_ollama_llm(cfg.LLM, format_schema=InvoiceSchema.model_json_schema())
```

and add `from rag.llm import get_ollama_llm` to app.py's imports (the Q&A path keeps `_get_ollama_llm()` — free-form answers must NOT be JSON-constrained).

- [ ] **Step 8: Run fast suite, then the slow retriever test**

Run: `python -m pytest -m "not slow" -q` — expected: all pass
Run: `python -m pytest tests/test_hybrid_retriever.py -m slow -q` — expected: pass

- [ ] **Step 9: Commit**

```bash
git add rag/hybrid_retriever.py rag/extractor.py rag/llm.py vision/gemini.py app.py tests/
git commit -m "feat: whole-document schema-constrained extraction"
```

---

### Task 3: Retrieval correctness + performance + agent retry fixes

Three architect findings: (a) BM25 tokenizes with bare `split()` so "Total:" never matches "total"; (b) the embedding model reloads from disk on every click; (c) the agent's rewrite-retry loop re-runs an identical prompt at temperature 0 — a provable no-op.

**Files:**
- Modify: `rag/hybrid_retriever.py`, `rag/agent.py`, `rag/ocr.py`, `app.py`, `config.yml`
- Test: `tests/test_hybrid_retriever.py`, `tests/test_agent.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_hybrid_retriever.py`:

```python
from rag.hybrid_retriever import _tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert _tokenize("Total: $212.09 DUE") == ["total", "212", "09", "due"]


def test_tokenize_empty():
    assert _tokenize("") == []
```

Append to `tests/test_agent.py`:

```python
class _RecordingLLM:
    def __init__(self, responses: dict):
        self._responses = responses
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        prompt_lower = prompt.lower()
        for keyword, response in self._responses.items():
            if keyword in prompt_lower:
                return response
        return "default response"


def test_rewrite_retry_includes_previous_attempt():
    llm = _RecordingLLM({
        "rewrite": "What is the invoice number?",
        "relevant": "no",                      # force retries
        "use the following": "not found",
        "supported": "yes",
    })
    retriever = _make_mock_retriever([{"text": "unrelated", "page": 1, "score": 0.1}])
    agent = build_agent(retriever, llm=llm)
    agent.invoke({
        "query": "invoice number?", "rewritten_query": "", "chunks": [],
        "answer": "", "relevant": False, "grounded": False,
        "iterations": 0, "critique_iterations": 0,
    })
    rewrite_prompts = [p for p in llm.prompts if "rewrite" in p.lower()]
    assert len(rewrite_prompts) >= 2
    assert "previous rewrite" in rewrite_prompts[1].lower()
```

- [ ] **Step 2: Run to verify failures**

Run: `python -m pytest tests/test_hybrid_retriever.py tests/test_agent.py -m "not slow" -q`
Expected: FAIL — `_tokenize` doesn't exist; "previous rewrite" not in second prompt

- [ ] **Step 3: Implement the tokenizer in rag/hybrid_retriever.py**

Add at module level (after imports, add `import re` to imports):

```python
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())
```

Use it for both index build and queries:
- In `__init__`: `self._bm25 = BM25Okapi([_tokenize(t) for t in self._texts])` — but BM25Okapi crashes on an all-empty corpus; keep behavior identical otherwise.
- In `retrieve`: `tokenized_query = _tokenize(query)`

Also accept injectable embeddings (for app-level caching):

```python
    def __init__(self, sha_key: str, base_dir: Path = Path("."), embeddings=None):
        ...
        if embeddings is not None:
            self._embeddings = embeddings
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=cfg.EMBEDDINGS,
                model_kwargs={"device": cfg.DEVICE},
                encode_kwargs={"normalize_embeddings": cfg.NORMALIZE_EMBEDDINGS},
            )
```

- [ ] **Step 4: Feedback-injected rewrite in rag/agent.py**

Replace `query_rewriter`:

```python
    def query_rewriter(state: AgentState) -> AgentState:
        prompt = (
            f"Rewrite this invoice question to be specific and extractable.\n"
            f"Original: {state['query']}\n"
        )
        if state.get("rewritten_query"):
            prompt += (
                f"A previous rewrite '{state['rewritten_query']}' retrieved irrelevant "
                f"context; produce a substantively different phrasing.\n"
            )
        prompt += "Rewritten:"
        rewritten = llm.invoke(prompt).strip()
        return {
            **state,
            "rewritten_query": rewritten,
            "iterations": state.get("iterations", 0) + 1,
        }
```

- [ ] **Step 5: Cache the RapidOCR engine in rag/ocr.py**

```python
from pathlib import Path

_ENGINE = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def ocr_pdf_pages(pdf_path: Path, scale: float = 2.0) -> list[str]:
    """Render each PDF page to an image and OCR it. Heavy imports deferred."""
    import numpy as np
    import pypdfium2 as pdfium

    engine = _get_engine()
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
```

- [ ] **Step 6: Cache embeddings in app.py**

Add near `_get_ollama_llm`:

```python
@st.cache_resource(show_spinner=False)
def _get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDINGS,
        model_kwargs={"device": cfg.DEVICE},
        encode_kwargs={"normalize_embeddings": cfg.NORMALIZE_EMBEDDINGS},
    )
```

Replace every `HybridRetriever(<key>, base_dir=BASE_DIR)` in app.py (Q&A, Extract, Compare — 3 sites) with `HybridRetriever(<key>, base_dir=BASE_DIR, embeddings=_get_embeddings())`.

- [ ] **Step 7: Clean config.yml**

Remove the dead `BM25_WEIGHT` / `DENSE_WEIGHT` lines (RRF never used them) and add `NUM_CTX: 8192` after `LLM:`. Then thread it: in app.py and rag/agent.py, wherever `get_ollama_llm(cfg.LLM, ...)` is called, pass `num_ctx=int(cfg.NUM_CTX)`; `_get_ollama_llm()` in app.py becomes:

```python
def _get_ollama_llm():
    from rag.llm import get_ollama_llm
    return get_ollama_llm(cfg.LLM, num_ctx=int(cfg.NUM_CTX))
```

and in rag/agent.py:

```python
    if llm is None:
        from rag.llm import get_ollama_llm
        llm = get_ollama_llm(cfg.LLM, num_ctx=int(cfg.NUM_CTX))
```

Also update the README config table: delete the BM25/DENSE weight rows if present, add `NUM_CTX | 8192 | Ollama context window`.

- [ ] **Step 8: Run the full suite (slow included — retrieval changed)**

Run: `python -m pytest -q`
Expected: all pass

- [ ] **Step 9: Commit**

```bash
git add rag/ app.py config.yml README.md tests/
git commit -m "fix: BM25 tokenization, cached embeddings/OCR engine, feedback-injected agent retries, num_ctx"
```

---

### Task 4: Extended schema + currency-aware comparison

Add the bookkeeping fields every real AP workflow needs (PO number, payment terms, vendor tax ID, addresses) and fix the comparator correctness bug: 100 USD vs 9000 INR flags a "discrepancy" while 100 USD vs 100 EUR passes.

**Files:**
- Modify: `models/invoice.py`, `rag/extractor.py` (prompt), `vision/gemini.py` (prompt), `rag/comparator.py`, `app.py` (`_schema_to_dfs` and metrics unchanged)
- Test: `tests/test_models.py`, `tests/test_comparator.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_models.py`:

```python
def test_extended_fields_default_none():
    s = InvoiceSchema()
    assert s.po_number is None
    assert s.payment_terms is None
    assert s.vendor_tax_id is None
    assert s.vendor_address is None
    assert s.bill_to is None
```

Append to `tests/test_comparator.py`:

```python
def test_same_amount_different_currency_not_total_flagged():
    a = _schema(total_amount=100.0, currency="USD")
    b = _schema(total_amount=9000.0, currency="INR")
    result = compare_invoices([("a", a), ("b", b)])
    fields = [d["field"] for d in result["discrepancies"]]
    assert "total_amount" not in fields       # cross-currency comparison is meaningless
    assert "currency" in fields               # but the mixed currencies are flagged


def test_total_mismatch_same_currency_still_flagged():
    a = _schema(total_amount=100.0, currency="USD")
    b = _schema(total_amount=200.0, currency="USD")
    result = compare_invoices([("a", a), ("b", b)])
    fields = [d["field"] for d in result["discrepancies"]]
    assert "total_amount" in fields


def test_po_number_in_table():
    a = _schema(po_number="PO-1")
    b = _schema(po_number="PO-2")
    result = compare_invoices([("a", a), ("b", b)])
    assert "po_number" in result["table"]
```

- [ ] **Step 2: Run to verify failures**

Run: `python -m pytest tests/test_models.py tests/test_comparator.py -q`
Expected: FAIL — pydantic unknown attribute / KeyError

- [ ] **Step 3: Extend models/invoice.py**

```python
class InvoiceSchema(BaseModel):
    vendor_name: str | None = None
    invoice_number: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    total_amount: float | None = None
    currency: str | None = None
    po_number: str | None = None
    payment_terms: str | None = None
    vendor_tax_id: str | None = None
    vendor_address: str | None = None
    bill_to: str | None = None
    line_items: list[LineItem] = []
```

- [ ] **Step 4: Update both extraction prompts**

In `rag/extractor.py` `_EXTRACTION_PROMPT`, add after the `"currency"` line (note doubled braces in this file):

```
  "po_number": string or null,
  "payment_terms": string or null,
  "vendor_tax_id": string or null,
  "vendor_address": string or null,
  "bill_to": string or null,
```

In `vision/gemini.py` `_EXTRACTION_PROMPT`, add the same five lines after `"currency"` (single braces in that file).

- [ ] **Step 5: Currency-aware comparator (rag/comparator.py)**

Update `_FIELDS` and the totals check:

```python
_FIELDS = [
    "vendor_name", "invoice_number", "invoice_date", "due_date",
    "subtotal", "tax", "total_amount", "currency", "po_number",
]
```

Replace the totals block:

```python
    currencies = {c.strip().upper() for c in table["currency"].values() if c and c.strip()}
    if len(currencies) > 1:
        discrepancies.append({
            "field": "currency",
            "detail": f"Mixed currencies — totals not compared: {', '.join(sorted(currencies))}",
        })

    totals = [(name, val) for name, val in table["total_amount"].items() if val is not None]
    if len(totals) >= 2 and len(currencies) <= 1:
        amounts = [v for _, v in totals]
        min_a, max_a = min(amounts), max(amounts)
        if min_a > 0 and (max_a - min_a) / min_a > 0.05:
            discrepancies.append({
                "field": "total_amount",
                "detail": f"Total mismatch >5%: {[f'{n}={v}' for n, v in totals]}",
            })
```

- [ ] **Step 6: Show the new fields in app.py `_schema_to_dfs`**

```python
def _schema_to_dfs(schema: InvoiceSchema):
    header = {
        "Field": ["Vendor", "Invoice #", "Date", "Due Date", "Subtotal", "Tax", "Total",
                  "Currency", "PO #", "Payment Terms", "Vendor Tax ID", "Vendor Address", "Bill To"],
        "Value": [
            schema.vendor_name, schema.invoice_number, schema.invoice_date,
            schema.due_date, schema.subtotal, schema.tax, schema.total_amount,
            schema.currency, schema.po_number, schema.payment_terms,
            schema.vendor_tax_id, schema.vendor_address, schema.bill_to,
        ],
    }
    line_items = [
        {"Description": li.description, "Qty": li.quantity, "Unit Price": li.unit_price, "Total": li.total}
        for li in schema.line_items
    ]
    return pd.DataFrame(header), pd.DataFrame(line_items) if line_items else pd.DataFrame()
```

- [ ] **Step 7: Run fast suite, commit**

Run: `python -m pytest -m "not slow" -q` — expected: all pass

```bash
git add models/invoice.py rag/extractor.py vision/gemini.py rag/comparator.py app.py tests/
git commit -m "feat: extended invoice schema; currency-aware comparison"
```

---

### Task 5: Persistent extractions + real exports

`schema_cache` dies on restart (the LLM must re-run tomorrow for the same invoice), the per-invoice CSV omits line items, and there's no combined export. Persist extraction JSON next to each invoice's store, rehydrate on discovery, and export a flat all-invoices CSV (one row per line item).

**Files:**
- Modify: `store.py`, `app.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write failing tests (append to tests/test_store.py)**

```python
from models.invoice import InvoiceSchema, LineItem
from store import save_extraction, load_extraction, all_extractions_dataframe


def _schema():
    return InvoiceSchema(
        vendor_name="ACME", invoice_number="A-1", total_amount=110.0, currency="USD",
        line_items=[LineItem(description="Widget", quantity=2, unit_price=50.0, total=100.0)],
    )


def test_save_and_load_pdf_extraction(tmp_path):
    _make_pdf_store(tmp_path, "abc12345", "acme.pdf")
    inv = discover_invoices(tmp_path)["abc12345"]
    save_extraction(inv, _schema(), tmp_path)
    loaded = load_extraction(inv, tmp_path)
    assert loaded is not None
    assert loaded.vendor_name == "ACME"
    assert loaded.line_items[0].total == 100.0


def test_discover_rehydrates_schema_cache(tmp_path):
    _make_pdf_store(tmp_path, "abc12345", "acme.pdf")
    inv = discover_invoices(tmp_path)["abc12345"]
    save_extraction(inv, _schema(), tmp_path)
    rediscovered = discover_invoices(tmp_path)["abc12345"]
    assert rediscovered["schema_cache"] is not None
    assert rediscovered["schema_cache"].invoice_number == "A-1"


def test_save_and_load_image_extraction(tmp_path):
    img_dir = tmp_path / "data" / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "receipt.png").write_bytes(b"fake")
    inv = discover_invoices(tmp_path)["img_receipt.png"]
    save_extraction(inv, _schema(), tmp_path)
    assert load_extraction(inv, tmp_path).vendor_name == "ACME"


def test_delete_image_removes_extraction_sidecar(tmp_path):
    img_dir = tmp_path / "data" / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "receipt.png").write_bytes(b"fake")
    inv = discover_invoices(tmp_path)["img_receipt.png"]
    save_extraction(inv, _schema(), tmp_path)
    delete_invoice(inv, tmp_path)
    assert not (img_dir / "receipt.png.extraction.json").exists()


def test_all_extractions_dataframe_one_row_per_line_item(tmp_path):
    invoices = {
        "k1": {"name": "a.pdf", "type": "pdf", "sha_key": "k1", "schema_cache": _schema()},
        "k2": {"name": "b.pdf", "type": "pdf", "sha_key": "k2", "schema_cache": None},
        "k3": {"name": "c.pdf", "type": "pdf", "sha_key": "k3",
               "schema_cache": InvoiceSchema(vendor_name="NoItems", total_amount=5.0)},
    }
    df = all_extractions_dataframe(invoices)
    assert len(df) == 2  # one line-item row for k1, one headers-only row for k3; k2 skipped
    assert set(df.columns) >= {"invoice", "vendor_name", "invoice_number", "total_amount",
                               "currency", "item_description", "item_quantity",
                               "item_unit_price", "item_total"}
    assert df.iloc[0]["item_description"] == "Widget"
```

- [ ] **Step 2: Run to verify failures**

Run: `python -m pytest tests/test_store.py -q`
Expected: FAIL — `ImportError: cannot import name 'save_extraction'`

- [ ] **Step 3: Implement in store.py (append; add imports)**

```python
import pandas as pd
from models.invoice import InvoiceSchema


def _extraction_path(inv: dict, base_dir: Path) -> Path:
    if inv["type"] == "pdf":
        return base_dir / "vectorstore" / inv["sha_key"] / "extraction.json"
    return Path(str(inv["path"]) + ".extraction.json")


def save_extraction(inv: dict, schema: InvoiceSchema, base_dir: Path) -> None:
    path = _extraction_path(inv, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(schema.model_dump_json(indent=2), encoding="utf8")


def load_extraction(inv: dict, base_dir: Path) -> InvoiceSchema | None:
    path = _extraction_path(inv, base_dir)
    if not path.exists():
        return None
    try:
        return InvoiceSchema.model_validate_json(path.read_text(encoding="utf8"))
    except Exception:
        return None  # stale/corrupt sidecar must not break discovery


def all_extractions_dataframe(invoices: dict) -> pd.DataFrame:
    header_fields = [
        "vendor_name", "invoice_number", "invoice_date", "due_date", "subtotal",
        "tax", "total_amount", "currency", "po_number", "payment_terms",
    ]
    rows = []
    for inv in invoices.values():
        schema = inv.get("schema_cache")
        if schema is None:
            continue
        base = {"invoice": inv["name"], **{f: getattr(schema, f) for f in header_fields}}
        if schema.line_items:
            for li in schema.line_items:
                rows.append({**base, "item_description": li.description,
                             "item_quantity": li.quantity, "item_unit_price": li.unit_price,
                             "item_total": li.total})
        else:
            rows.append({**base, "item_description": None, "item_quantity": None,
                         "item_unit_price": None, "item_total": None})
    return pd.DataFrame(rows)
```

Then wire persistence into the existing functions:
- `discover_invoices`: after building each entry (both pdf and image), set `entry["schema_cache"] = load_extraction(entry, base_dir)`. Easiest: build the dict first, then loop `for inv in invoices.values(): inv["schema_cache"] = load_extraction(inv, base_dir)` before returning.
- `delete_invoice` image branch: also `Path(str(inv["path"]) + ".extraction.json").unlink(missing_ok=True)` (the pdf branch already removes the whole vectorstore dir).

- [ ] **Step 4: Run store tests**

Run: `python -m pytest tests/test_store.py -q`
Expected: all pass (12 tests)

- [ ] **Step 5: Wire into app.py**

After every successful extraction (Extract tab try-block and Compare loop), add:

```python
                save_extraction(invoices[<the key>], schema, BASE_DIR)
```

(`<the key>` is `selected_key_ext` in the Extract tab, `key` in the Compare loop. Add `save_extraction` to the `from store import ...` line.)

Per-invoice CSV with line items — replace the CSV download in the Extract tab:

```python
            col2.download_button(
                "Download CSV (incl. line items)",
                data=all_extractions_dataframe(
                    {selected_key_ext: invoices[selected_key_ext]}
                ).to_csv(index=False),
                file_name=f"{selected_ext['name']}_extracted.csv",
                mime="text/csv",
            )
```

Combined export — in the sidebar after the "Loaded invoices" list, add:

```python
    extracted = {k: v for k, v in st.session_state["invoices"].items() if v.get("schema_cache")}
    if extracted:
        st.download_button(
            "Download all extractions (CSV)",
            data=all_extractions_dataframe(extracted).to_csv(index=False),
            file_name="all_invoices.csv",
            mime="text/csv",
            use_container_width=True,
        )
```

(Add `all_extractions_dataframe` to the store import.)

- [ ] **Step 6: Run fast suite, commit**

Run: `python -m pytest -m "not slow" -q` — expected: all pass

```bash
git add store.py app.py tests/test_store.py
git commit -m "feat: persist extractions to disk; line-item CSV and all-invoices export"
```

---

### Task 6: Synthetic eval harness with published accuracy

Nobody can currently say whether any of this works. Generate 12 deterministic synthetic invoices with known ground truth, run the local extraction pipeline against them, score per-field accuracy, and publish the table in the README. Fully offline and reproducible.

**Files:**
- Create: `eval/__init__.py` (empty), `eval/generate_dataset.py`, `eval/scoring.py`, `eval/run_eval.py`
- Test: `tests/test_eval.py`
- Modify: `requirements.txt` (fpdf2), `README.md` (results table), `.gitignore` (eval artifacts)

- [ ] **Step 1: Install fpdf2 and add to requirements**

Run: `python -m pip install --no-cache-dir -q fpdf2` then `python -c "from fpdf import FPDF; print('fpdf2 OK')"`
Append `fpdf2>=2.7.0` to `requirements.txt`.
**Check the numpy pin survived:** `python -c "import numpy; assert numpy.__version__.startswith('1.26'), numpy.__version__; print('numpy OK')"`

- [ ] **Step 2: Write failing tests**

```python
# tests/test_eval.py
import json
import pytest
from pathlib import Path

from eval.generate_dataset import generate_dataset
from eval.scoring import field_match, score_invoice
from models.invoice import InvoiceSchema
from rag.validator import validate_invoice


def test_generate_dataset_creates_labeled_pairs(tmp_path):
    pairs = generate_dataset(tmp_path, n=3, seed=7)
    assert len(pairs) == 3
    for pdf_path, truth_path in pairs:
        assert pdf_path.exists() and pdf_path.suffix == ".pdf"
        truth = InvoiceSchema.model_validate_json(truth_path.read_text(encoding="utf8"))
        assert truth.vendor_name and truth.invoice_number
        assert validate_invoice(truth) == []   # ground truth must be arithmetically consistent


def test_generate_dataset_deterministic(tmp_path):
    a = generate_dataset(tmp_path / "a", n=2, seed=42)
    b = generate_dataset(tmp_path / "b", n=2, seed=42)
    truth_a = a[0][1].read_text(encoding="utf8")
    truth_b = b[0][1].read_text(encoding="utf8")
    assert truth_a == truth_b


def test_field_match_strings_case_insensitive():
    assert field_match("ACME Corp", "acme corp") is True
    assert field_match("ACME Corp", "Beta Ltd") is False


def test_field_match_numbers_tolerant():
    assert field_match(212.09, 212.10) is True
    assert field_match(212.09, 250.0) is False


def test_field_match_none_handling():
    assert field_match(None, None) is True
    assert field_match(None, "x") is False


def test_score_invoice_perfect():
    truth = InvoiceSchema(vendor_name="A", invoice_number="1", total_amount=10.0, currency="USD")
    scores = score_invoice(truth, truth)
    assert all(v == 1.0 for v in scores.values())
```

- [ ] **Step 3: Run to verify failures**

Run: `python -m pytest tests/test_eval.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.generate_dataset'`

- [ ] **Step 4: Implement eval/generate_dataset.py (and create empty eval/__init__.py)**

```python
# eval/generate_dataset.py
"""Generate deterministic synthetic invoices (PDF + ground-truth JSON)."""
import json
import random
from pathlib import Path

from fpdf import FPDF

from models.invoice import InvoiceSchema, LineItem

_VENDORS = [
    ("Hartley Office Supply", "HOS-2291-T", "Net 30"),
    ("Brightline Logistics", "BLL-8830-X", "Net 15"),
    ("Cascade IT Services", "CIS-1104-R", "Due on receipt"),
    ("Meridian Catering Co", "MCC-5512-B", "Net 45"),
]
_ITEMS = [
    ("A4 paper ream 80gsm", 4.25), ("Wireless keyboard", 38.90),
    ("Server rack shelf", 112.50), ("Catering lunch tray", 14.75),
    ("Network patch cable 3m", 6.40), ("Monitor stand dual", 54.20),
    ("Coffee beans 1kg", 19.80), ("Whiteboard markers x10", 8.95),
]


def _make_truth(rng: random.Random, idx: int) -> InvoiceSchema:
    vendor, tax_id, terms = rng.choice(_VENDORS)
    items = []
    for _ in range(rng.randint(2, 4)):
        desc, price = rng.choice(_ITEMS)
        qty = rng.randint(1, 6)
        items.append(LineItem(description=desc, quantity=float(qty),
                              unit_price=price, total=round(qty * price, 2)))
    subtotal = round(sum(li.total for li in items), 2)
    tax = round(subtotal * 0.10, 2)
    return InvoiceSchema(
        vendor_name=vendor,
        invoice_number=f"INV-{2026}{idx:04d}",
        invoice_date=f"2026-{rng.randint(1, 6):02d}-{rng.randint(1, 28):02d}",
        due_date=None,
        subtotal=subtotal, tax=tax, total_amount=round(subtotal + tax, 2),
        currency="USD", po_number=f"PO-{rng.randint(1000, 9999)}",
        payment_terms=terms, vendor_tax_id=tax_id,
        line_items=items,
    )


def _render_pdf(truth: InvoiceSchema, path: Path) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "INVOICE", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"Vendor: {truth.vendor_name}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Tax ID: {truth.vendor_tax_id}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Invoice Number: {truth.invoice_number}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Invoice Date: {truth.invoice_date}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"PO Number: {truth.po_number}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Payment Terms: {truth.payment_terms}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(90, 7, "Description"); pdf.cell(25, 7, "Qty"); pdf.cell(35, 7, "Unit Price")
    pdf.cell(30, 7, "Total", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for li in truth.line_items:
        pdf.cell(90, 7, li.description); pdf.cell(25, 7, f"{li.quantity:.0f}")
        pdf.cell(35, 7, f"{li.unit_price:.2f}")
        pdf.cell(30, 7, f"{li.total:.2f}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.cell(0, 7, f"Subtotal: {truth.subtotal:.2f} USD", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Tax (10%): {truth.tax:.2f} USD", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Total Due: {truth.total_amount:.2f} USD", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


def generate_dataset(out_dir: Path, n: int = 12, seed: int = 42) -> list[tuple[Path, Path]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pairs = []
    for i in range(1, n + 1):
        truth = _make_truth(rng, i)
        pdf_path = out_dir / f"inv_{i:02d}.pdf"
        truth_path = out_dir / f"inv_{i:02d}.json"
        _render_pdf(truth, pdf_path)
        truth_path.write_text(truth.model_dump_json(indent=2), encoding="utf8")
        pairs.append((pdf_path, truth_path))
    return pairs


if __name__ == "__main__":
    pairs = generate_dataset(Path(__file__).parent / "dataset")
    print(f"generated {len(pairs)} invoices in eval/dataset/")
```

- [ ] **Step 5: Implement eval/scoring.py**

```python
# eval/scoring.py
from models.invoice import InvoiceSchema

_HEADER_FIELDS = [
    "vendor_name", "invoice_number", "invoice_date", "subtotal", "tax",
    "total_amount", "currency", "po_number", "payment_terms", "vendor_tax_id",
]


def field_match(expected, actual) -> bool:
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= max(abs(float(expected)) * 0.01, 0.011)
    return str(expected).strip().lower() == str(actual).strip().lower()


def score_invoice(truth: InvoiceSchema, predicted: InvoiceSchema) -> dict[str, float]:
    scores = {
        f: 1.0 if field_match(getattr(truth, f), getattr(predicted, f)) else 0.0
        for f in _HEADER_FIELDS
    }
    if truth.line_items:
        n_expected = len(truth.line_items)
        matched = 0
        remaining = list(predicted.line_items)
        for t in truth.line_items:
            for p in remaining:
                if field_match(t.total, p.total) and field_match(t.quantity, p.quantity):
                    matched += 1
                    remaining.remove(p)
                    break
        scores["line_items"] = matched / n_expected
    return scores
```

- [ ] **Step 6: Implement eval/run_eval.py**

```python
# eval/run_eval.py
"""Run extraction over the synthetic dataset and report per-field accuracy.

Requires Ollama running with the configured model. Usage:
    python -m eval.run_eval [--n 12]
"""
import argparse
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from eval.generate_dataset import generate_dataset
from eval.scoring import score_invoice, _HEADER_FIELDS
from ingest import ingest_pdf
from models.invoice import InvoiceSchema
from rag.extractor import extract_invoice, ExtractionError
from rag.hybrid_retriever import HybridRetriever
from rag.llm import get_ollama_llm
from rag.utils import load_config


def main(n: int = 12) -> Path:
    cfg = load_config()
    dataset_dir = Path(__file__).parent / "dataset"
    pairs = generate_dataset(dataset_dir, n=n)
    work = Path(tempfile.mkdtemp(prefix="eval_"))
    llm = get_ollama_llm(cfg.LLM, format_schema=InvoiceSchema.model_json_schema(),
                         num_ctx=int(cfg.NUM_CTX))

    totals: dict[str, list[float]] = defaultdict(list)
    failures = 0
    for pdf_path, truth_path in pairs:
        truth = InvoiceSchema.model_validate_json(truth_path.read_text(encoding="utf8"))
        sha = ingest_pdf(pdf_path, base_dir=work)
        retriever = HybridRetriever(sha, base_dir=work)
        try:
            predicted = extract_invoice(retriever, llm)
        except ExtractionError:
            failures += 1
            predicted = InvoiceSchema()
        for field, score in score_invoice(truth, predicted).items():
            totals[field].append(score)
        print(f"  {pdf_path.name}: done")

    lines = ["# Extraction Eval Results", "",
             f"Model: `{cfg.LLM}` (schema-constrained, whole-document) — "
             f"{n} synthetic invoices, {failures} hard failures", "",
             "| Field | Accuracy |", "|---|---|"]
    for field in _HEADER_FIELDS + ["line_items"]:
        if field in totals:
            acc = sum(totals[field]) / len(totals[field])
            lines.append(f"| {field} | {acc:.0%} |")
    overall = sum(sum(v) for v in totals.values()) / sum(len(v) for v in totals.values())
    lines += ["", f"**Overall field accuracy: {overall:.0%}**"]

    out = Path(__file__).parent / "results.md"
    out.write_text("\n".join(lines), encoding="utf8")
    shutil.rmtree(work, ignore_errors=True)
    print(f"\nwrote {out}\noverall: {overall:.0%}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    main(parser.parse_args().n)
```

- [ ] **Step 7: Run unit tests**

Run: `python -m pytest tests/test_eval.py -q`
Expected: 7 passed

- [ ] **Step 8: Run the real eval (Ollama must be up; ~5–15 min on CPU)**

Run: `python -m eval.run_eval`
Expected: per-invoice progress lines, then `wrote eval\results.md` with an overall percentage. If the overall is implausibly low (<40%), STOP and investigate (likely the `format_schema` fallback from Task 2 Step 4 — check whether `llm.format` is actually constraining).

- [ ] **Step 9: Publish in README + ignore dataset artifacts**

Append `eval/dataset/` to `.gitignore`. In `README.md`, after the "Features" section, add (replace the placeholders with the ACTUAL numbers from eval/results.md — this is runtime data, paste what the run produced):

```markdown
## Extraction Accuracy

Measured on 12 synthetic labeled invoices (deterministic, regenerate with
`python -m eval.generate_dataset`; score with `python -m eval.run_eval`):

| Field | Accuracy |
|---|---|
| vendor_name | <paste>% |
| invoice_number | <paste>% |
| total_amount | <paste>% |
| line_items | <paste>% |

**Overall field accuracy: <paste>%** — `llama3.2:3b`, schema-constrained
decoding, whole-document context, fully offline.
```

(Include all field rows from results.md, not just these four.)

- [ ] **Step 10: Commit**

```bash
git add eval/ tests/test_eval.py requirements.txt README.md .gitignore
git commit -m "feat: synthetic eval harness with published extraction accuracy"
```

---

### Task 7: Document preview + inline correction

Users can't see the invoice next to the extraction and can't fix a wrong field — the two biggest trust killers. Render page 1 next to the results and make the tables editable with an "Apply corrections" button that persists.

**Files:**
- Create: `rag/preview.py`
- Modify: `store.py` (move `schema_to_dfs` here, add `schema_from_dfs`), `app.py`
- Test: `tests/test_preview.py`, `tests/test_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preview.py
from rag.preview import render_pdf_page


def test_render_pdf_page_returns_image(invoice_pdf):
    img = render_pdf_page(invoice_pdf)
    assert img.width > 100 and img.height > 100
```

Append to `tests/test_store.py`:

```python
from store import schema_to_dfs, schema_from_dfs


def test_schema_dfs_round_trip():
    original = InvoiceSchema(
        vendor_name="ACME", invoice_number="A-1", subtotal=100.0, tax=10.0,
        total_amount=110.0, currency="USD", po_number="PO-7",
        line_items=[LineItem(description="Widget", quantity=2.0, unit_price=50.0, total=100.0)],
    )
    header_df, items_df = schema_to_dfs(original)
    rebuilt = schema_from_dfs(header_df, items_df)
    assert rebuilt == original


def test_schema_from_dfs_coerces_numeric_strings():
    original = InvoiceSchema(total_amount=110.0)
    header_df, items_df = schema_to_dfs(original)
    header_df.loc[header_df["Field"] == "Total", "Value"] = "212.09"
    rebuilt = schema_from_dfs(header_df, items_df)
    assert rebuilt.total_amount == 212.09


def test_schema_from_dfs_blank_strings_become_none():
    original = InvoiceSchema(vendor_name="ACME")
    header_df, items_df = schema_to_dfs(original)
    header_df.loc[header_df["Field"] == "Vendor", "Value"] = ""
    rebuilt = schema_from_dfs(header_df, items_df)
    assert rebuilt.vendor_name is None
```

- [ ] **Step 2: Run to verify failures**

Run: `python -m pytest tests/test_preview.py tests/test_store.py -m "not slow" -q`
Expected: FAIL — no module `rag.preview`; cannot import `schema_to_dfs` from store

- [ ] **Step 3: Implement rag/preview.py**

```python
# rag/preview.py
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
```

- [ ] **Step 4: Move schema_to_dfs into store.py and add schema_from_dfs**

In `store.py` (append; `pd` and `InvoiceSchema` already imported from Task 5; add `from models.invoice import LineItem`):

```python
_HEADER_ROWS: list[tuple[str, str, bool]] = [
    # (display label, schema field, is_numeric)
    ("Vendor", "vendor_name", False), ("Invoice #", "invoice_number", False),
    ("Date", "invoice_date", False), ("Due Date", "due_date", False),
    ("Subtotal", "subtotal", True), ("Tax", "tax", True), ("Total", "total_amount", True),
    ("Currency", "currency", False), ("PO #", "po_number", False),
    ("Payment Terms", "payment_terms", False), ("Vendor Tax ID", "vendor_tax_id", False),
    ("Vendor Address", "vendor_address", False), ("Bill To", "bill_to", False),
]


def schema_to_dfs(schema: InvoiceSchema):
    header = {
        "Field": [label for label, _, _ in _HEADER_ROWS],
        "Value": [getattr(schema, field) for _, field, _ in _HEADER_ROWS],
    }
    line_items = [
        {"Description": li.description, "Qty": li.quantity,
         "Unit Price": li.unit_price, "Total": li.total}
        for li in schema.line_items
    ]
    return pd.DataFrame(header), pd.DataFrame(line_items) if line_items else pd.DataFrame()


def _coerce(value, numeric: bool):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if numeric:
        return float(str(value).replace(",", ""))
    return str(value)


def schema_from_dfs(header_df: pd.DataFrame, items_df: pd.DataFrame) -> InvoiceSchema:
    values = dict(zip(header_df["Field"], header_df["Value"]))
    fields = {
        field: _coerce(values.get(label), numeric)
        for label, field, numeric in _HEADER_ROWS
    }
    items = []
    if not items_df.empty:
        for _, row in items_df.iterrows():
            desc = _coerce(row.get("Description"), False)
            if desc is None:
                continue
            items.append(LineItem(
                description=desc,
                quantity=_coerce(row.get("Qty"), True),
                unit_price=_coerce(row.get("Unit Price"), True),
                total=_coerce(row.get("Total"), True),
            ))
    return InvoiceSchema(**fields, line_items=items)
```

In `app.py`: delete the local `_schema_to_dfs` function, add `schema_to_dfs, schema_from_dfs` to the store import, and rename the two call sites from `_schema_to_dfs(` to `schema_to_dfs(`.

- [ ] **Step 5: Run the new tests**

Run: `python -m pytest tests/test_preview.py tests/test_store.py -m "not slow" -q`
Expected: all pass

- [ ] **Step 6: Preview + editor UI in app.py Extract tab**

Replace the `if cached:` display block (metrics stay; tables become editable inside a two-column layout):

```python
        cached = invoices[selected_key_ext].get("schema_cache")
        if cached:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total", f"{cached.total_amount:,.2f}" if cached.total_amount is not None else "—")
            m2.metric("Vendor", cached.vendor_name or "—")
            m3.metric("Invoice #", cached.invoice_number or "—")
            m4.metric("Date", cached.invoice_date or "—")

            from rag.validator import has_amounts
            checks = validate_invoice(cached)
            if checks:
                st.subheader("Validation checks")
                for w in checks:
                    st.warning(w)
            elif has_amounts(cached):
                st.caption("All arithmetic checks pass.")
            else:
                st.warning("No monetary fields were extracted — treat this result as incomplete.")

            preview_col, data_col = st.columns([2, 3])
            with preview_col:
                st.subheader("Document")
                try:
                    if selected_ext["type"] == "image":
                        st.image(str(selected_ext["path"]), use_container_width=True)
                    else:
                        pdf_dir = BASE_DIR / "data" / selected_ext["sha_key"]
                        pdfs = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
                        if pdfs:
                            from rag.preview import render_pdf_page
                            st.image(render_pdf_page(pdfs[0]), use_container_width=True)
                        else:
                            st.caption("Source PDF not found on disk.")
                except Exception as e:
                    st.caption(f"Preview unavailable: {e}")

            with data_col:
                header_df, items_df = schema_to_dfs(cached)
                st.subheader("Header fields")
                edited_header = st.data_editor(header_df, use_container_width=True,
                                               hide_index=True, key="edit_header",
                                               disabled=["Field"])
                st.subheader("Line items")
                edited_items = st.data_editor(items_df, use_container_width=True,
                                              hide_index=True, key="edit_items",
                                              num_rows="dynamic")
                if st.button("Apply corrections", key="apply_corrections"):
                    try:
                        corrected = schema_from_dfs(edited_header, edited_items)
                        invoices[selected_key_ext]["schema_cache"] = corrected
                        save_extraction(invoices[selected_key_ext], corrected, BASE_DIR)
                        st.success("Corrections saved.")
                        st.rerun()
                    except (ValueError, TypeError) as e:
                        st.error(f"Could not apply corrections: {e}")

            col1, col2 = st.columns(2)
            col1.download_button(
                "Download JSON",
                data=cached.model_dump_json(indent=2),
                file_name=f"{selected_ext['name']}_extracted.json",
                mime="application/json",
            )
            col2.download_button(
                "Download CSV (incl. line items)",
                data=all_extractions_dataframe(
                    {selected_key_ext: invoices[selected_key_ext]}
                ).to_csv(index=False),
                file_name=f"{selected_ext['name']}_extracted.csv",
                mime="text/csv",
            )
```

(Note: `st.data_editor` on an empty `items_df` with `num_rows="dynamic"` lets users add line items from scratch — intended.)

- [ ] **Step 7: Run fast suite, commit**

Run: `python -m pytest -m "not slow" -q` — expected: all pass

```bash
git add rag/preview.py store.py app.py tests/test_preview.py tests/test_store.py
git commit -m "feat: document preview and inline correction with persistence"
```

---

### Task 8: Batch upload + Extract All

One file per upload-click doesn't survive contact with a stack of invoices.

**Files:**
- Modify: `app.py`
- Test: existing `tests/test_app.py` must stay green (file-uploader interactions aren't scriptable in AppTest; verification is the suite + a later live check)

- [ ] **Step 1: Multi-file uploader and batch ingest loop in app.py sidebar**

Replace the uploader and the `if uploaded and add_clicked:` block:

```python
    uploaded_files = st.file_uploader(
        "PDF or Image", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True
    )

    st.markdown('<p class="step-label">STEP 2 · LOAD THEM</p>', unsafe_allow_html=True)
    add_clicked = st.button(
        "Add Invoices", type="primary", use_container_width=True,
        disabled=not uploaded_files,
    )
    if not uploaded_files:
        st.caption("The button unlocks once files are chosen.")

    if uploaded_files and add_clicked:
        progress = st.progress(0.0, text="Starting…")
        ok, failed = 0, 0
        for i, uploaded in enumerate(uploaded_files, 1):
            progress.progress(i / len(uploaded_files), text=f"Processing {uploaded.name}…")
            suffix = Path(uploaded.name).suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)
            try:
                if suffix == ".pdf":
                    sha_key = ingest_pdf(
                        tmp_path, base_dir=BASE_DIR,
                        original_name=_safe_filename(uploaded.name, suffix),
                    )
                    st.session_state["invoices"][sha_key] = {
                        "name": uploaded.name, "type": "pdf",
                        "sha_key": sha_key, "schema_cache": None,
                    }
                else:
                    img_dest = BASE_DIR / "data" / "images"
                    img_dest.mkdir(parents=True, exist_ok=True)
                    safe_name = _safe_filename(uploaded.name, suffix)
                    dest = img_dest / safe_name
                    dest.write_bytes(uploaded.getvalue())
                    st.session_state["invoices"][f"img_{safe_name}"] = {
                        "name": safe_name, "type": "image",
                        "path": dest, "schema_cache": None,
                    }
                ok += 1
            except Exception as e:
                failed += 1
                st.error(f"{uploaded.name}: {e}")
            finally:
                tmp_path.unlink(missing_ok=True)
        progress.empty()
        if ok:
            st.success(f"Loaded {ok} invoice(s)." + (f" {failed} failed." if failed else ""))
```

Also update the sidebar step-1 label text from `STEP 1 · CHOOSE A FILE` to `STEP 1 · CHOOSE FILES`.

- [ ] **Step 2: "Extract all" button in the Extract tab**

After the per-invoice "Extract All Fields" button block, add:

```python
        pending = {k: v for k, v in invoices.items() if v.get("schema_cache") is None}
        if len(pending) > 1 and st.button(f"Extract all {len(pending)} pending invoices", key="extract_all_btn"):
            from rag.extractor import ExtractionError
            progress = st.progress(0.0)
            done, errors = 0, 0
            for i, (key, inv) in enumerate(pending.items(), 1):
                progress.progress(i / len(pending), text=f"Extracting {inv['name']}…")
                try:
                    if inv["type"] == "pdf":
                        retriever = HybridRetriever(inv["sha_key"], base_dir=BASE_DIR,
                                                    embeddings=_get_embeddings())
                        llm = get_ollama_llm(cfg.LLM, format_schema=InvoiceSchema.model_json_schema(),
                                             num_ctx=int(cfg.NUM_CTX))
                        schema = extract_invoice(retriever, llm)
                    else:
                        schema = extract_invoice_gemini(inv["path"])
                    invoices[key]["schema_cache"] = schema
                    save_extraction(invoices[key], schema, BASE_DIR)
                    done += 1
                except Exception as e:
                    errors += 1
                    st.error(f"{inv['name']}: {e}")
            progress.empty()
            st.success(f"Extracted {done}." + (f" {errors} failed." if errors else ""))
            st.rerun()
```

- [ ] **Step 3: Run fast suite (AppTest must still pass), commit**

Run: `python -m pytest -m "not slow" -q` — expected: all pass

```bash
git add app.py
git commit -m "feat: batch upload with progress; extract-all for pending invoices"
```

---

### Task 9: Dashboard tab + final verification

Cross-invoice visibility: a sortable register of all extracted invoices, spend by vendor, monthly totals — grouped by currency, never summed across currencies.

**Files:**
- Modify: `app.py`, `tests/test_app.py`, `README.md`
- Test: `tests/test_app.py`

- [ ] **Step 1: Update the tab-count tests (now four tabs)**

In `tests/test_app.py`, rename/replace `test_app_has_three_tabs` and fix the auth test:

```python
def test_app_has_four_tabs():
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert len(at.tabs) == 4
```

and in `test_app_open_when_password_unset`, change the assertion to `assert len(at.tabs) == 4`.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_app.py -q`
Expected: FAIL — 3 tabs found, 4 expected

- [ ] **Step 3: Add the Dashboard tab in app.py**

Change the tab declaration:

```python
qa_tab, extract_tab, compare_tab, dash_tab = st.tabs(["Q&A", "Extract", "Compare", "Dashboard"])
```

Append at the end of the file:

```python
# ── Dashboard Tab ─────────────────────────────────────────────────────────────
with dash_tab:
    extracted = {k: v for k, v in invoices.items() if v.get("schema_cache")}
    if not extracted:
        st.markdown(
            """
            <div class="empty-state">
              <h3>No extractions yet</h3>
              <p>Extract at least one invoice and it will show up here —
              a register of all invoices, spend by vendor, and monthly totals.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        df = all_extractions_dataframe(extracted)
        register = df.drop_duplicates(subset=["invoice"])[
            ["invoice", "vendor_name", "invoice_number", "invoice_date",
             "due_date", "total_amount", "currency"]
        ]
        st.subheader("Invoice register")
        st.dataframe(register, use_container_width=True, hide_index=True)

        from rag.comparator import _parse_date
        for currency, group in register.groupby(register["currency"].fillna("?")):
            valid = group.dropna(subset=["total_amount"])
            if valid.empty:
                continue
            st.subheader(f"Spend by vendor ({currency})")
            by_vendor = valid.groupby(valid["vendor_name"].fillna("Unknown"))["total_amount"].sum()
            st.bar_chart(by_vendor)

            dated = valid.assign(_d=valid["invoice_date"].map(_parse_date)).dropna(subset=["_d"])
            if len(dated) >= 2:
                st.subheader(f"Monthly totals ({currency})")
                monthly = dated.assign(month=dated["_d"].dt.to_period("M").astype(str)) \
                               .groupby("month")["total_amount"].sum()
                st.line_chart(monthly)
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest -q`
Expected: all pass

- [ ] **Step 5: Live verification (run skill pattern)**

Start the app (`python -m streamlit run app.py --server.headless true --server.port 8501` in background), then with Playwright (headless chromium, pattern from earlier sessions): upload 2 eval-dataset PDFs in one batch, Extract all, check the Dashboard renders the register + bar chart, screenshot, LOOK at the screenshot. Stop the server after. Fix anything broken before committing.

- [ ] **Step 6: Update README features list**

Add to the Features section: batch upload, document preview with inline correction, persistent extractions, all-invoices CSV export, analytics dashboard, published eval accuracy.

- [ ] **Step 7: Commit, push, watch CI**

```bash
git add app.py tests/test_app.py README.md
git commit -m "feat: analytics dashboard tab (register, spend by vendor, monthly totals)"
git push origin master
gh run watch $(gh run list --repo Amanbatra03/invoice-extractor --limit 1 --json databaseId --jq '.[0].databaseId') --repo Amanbatra03/invoice-extractor --exit-status
```

Expected: CI green.

---

## Self-Review Notes

- **Spec coverage:** failure visibility (T1), whole-doc + schema-constrained (T2), retrieval/perf/agent fixes (T3), extended schema + currency-aware compare (T4), persistence + exports (T5), eval harness + published numbers (T6), preview + correction (T7), batch upload (T8), dashboard (T9). Hosted demo deliberately excluded — requires the user's hosting accounts; flagged as follow-up.
- **Type consistency:** `get_ollama_llm(model, temperature=0, format_schema=None, num_ctx=8192)` used identically in T2/T3/T6/T8. `schema_to_dfs`/`schema_from_dfs` defined T7, used T7+T5's export remains separate. `all_extractions_dataframe(invoices: dict)` consistent T5/T9. `ExtractionError` defined T1, imported T2/T6/T8. `_HEADER_ROWS` numeric flags align with float fields in `models/invoice.py`.
- **Known risk points called out inline:** langchain-ollama dict-`format` support (T2 Step 4 fallback), low eval score stop-rule (T6 Step 8), numpy pin check after fpdf2 install (T6 Step 1).
- **Ordering:** T6 runs after T2/T4 so the published numbers reflect the final extraction pipeline. T7/T9 depend on T5's `save_extraction`/`all_extractions_dataframe`.
