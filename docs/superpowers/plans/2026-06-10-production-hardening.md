# Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six highest-value production gaps in invoice-extractor: CI, real deletion + invoice rediscovery, OCR fallback for scanned PDFs, extraction validation, basic auth, and Docker packaging.

**Architecture:** Each gap is an independent, shippable task. New cross-cutting logic goes in small focused modules (`store.py`, `rag/ocr.py`, `rag/validator.py`, `rag/llm.py`, `auth.py`) so `app.py` stays a thin UI layer. TDD throughout; fast tests must not import torch/chromadb at module scope (lazy imports, established pattern in this repo).

**Tech Stack:** Streamlit, pytest, GitHub Actions, pypdfium2 + rapidocr-onnxruntime (OCR, no system deps), Docker + compose with an Ollama sidecar.

**Environment notes (this machine):**
- Windows 11, anaconda Python 3.11.9; run commands from `C:\Users\amanb\invoice-extractor`
- This machine fails to load native wheels built with newest MSVC toolchains (WinError 1114). Working pins already installed: torch 2.7.1+cpu, chromadb 0.6.3, onnxruntime 1.19.2. When installing rapidocr, pin onnxruntime: `pip install rapidocr-onnxruntime "onnxruntime==1.19.2"`
- Full suite: `python -m pytest -q` (~75s). Fast only: `python -m pytest -m "not slow" -q`

---

## File Structure

```
.github/workflows/ci.yml      # NEW — fast-test CI on push/PR
store.py                      # NEW — discover_invoices, delete_invoice (filesystem lifecycle)
rag/ocr.py                    # NEW — ocr_pdf_pages (pypdfium2 render + RapidOCR)
rag/validator.py              # NEW — validate_invoice arithmetic checks
rag/llm.py                    # NEW — get_ollama_llm honoring OLLAMA_BASE_URL
auth.py                       # NEW — check_password gate (APP_PASSWORD env)
Dockerfile, docker-compose.yml, .dockerignore   # NEW
ingest.py                     # MODIFY — page-text extraction with OCR fallback
rag/agent.py                  # MODIFY — use rag.llm factory
app.py                        # MODIFY — wire store/auth/validator/llm factory
requirements.txt              # MODIFY — add pypdfium2, rapidocr-onnxruntime, onnxruntime
tests/test_store.py           # NEW
tests/test_ocr.py             # NEW
tests/test_validator.py       # NEW
tests/test_llm.py             # NEW
tests/test_app.py             # MODIFY — auth gate tests
```

---

### Task 1: GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**

```yaml
name: CI

on:
  push:
    branches: [master, main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip
      - name: Install CPU torch first (avoids 2GB CUDA wheels)
        run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run fast tests
        run: pytest -m "not slow" -q
```

- [ ] **Step 2: Validate the YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('valid')"`
Expected: `valid`

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: run fast test suite on push and PR"
git push origin master
```

- [ ] **Step 4: Verify the run on GitHub**

Run: `gh run watch --repo Amanbatra03/invoice-extractor --exit-status` (or `gh run list --limit 1` until status is `completed success`)
Expected: CI run passes. If it fails, read the log with `gh run view --log-failed` and fix before proceeding.

---

### Task 2: Real delete + invoice rediscovery on startup

Deleting an invoice currently only removes the session entry — the PDF copy and vectorstore stay on disk forever (a PII-retention bug). And on app restart the session list is empty even though indexes exist on disk.

**Files:**
- Create: `store.py`
- Test: `tests/test_store.py`
- Modify: `app.py` (session init + delete handler)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_store.py
import json
from pathlib import Path

from store import discover_invoices, delete_invoice


def _make_pdf_store(base: Path, sha: str, pdf_name: str):
    vs = base / "vectorstore" / sha
    vs.mkdir(parents=True)
    (vs / "bm25.json").write_text(json.dumps({"texts": ["x"], "pages": [1]}))
    d = base / "data" / sha
    d.mkdir(parents=True)
    (d / pdf_name).write_bytes(b"%PDF-1.4 fake")


def test_discover_finds_pdf_invoices(tmp_path):
    _make_pdf_store(tmp_path, "abc12345", "acme_march.pdf")
    invoices = discover_invoices(tmp_path)
    assert "abc12345" in invoices
    assert invoices["abc12345"]["name"] == "acme_march.pdf"
    assert invoices["abc12345"]["type"] == "pdf"
    assert invoices["abc12345"]["sha_key"] == "abc12345"
    assert invoices["abc12345"]["schema_cache"] is None


def test_discover_skips_partial_vectorstore(tmp_path):
    (tmp_path / "vectorstore" / "deadbeef").mkdir(parents=True)  # no bm25.json
    assert discover_invoices(tmp_path) == {}


def test_discover_finds_images(tmp_path):
    img_dir = tmp_path / "data" / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "receipt.png").write_bytes(b"fake")
    invoices = discover_invoices(tmp_path)
    assert "img_receipt.png" in invoices
    assert invoices["img_receipt.png"]["type"] == "image"


def test_discover_empty_dir(tmp_path):
    assert discover_invoices(tmp_path) == {}


def test_delete_pdf_removes_all_files(tmp_path):
    _make_pdf_store(tmp_path, "abc12345", "acme.pdf")
    inv = discover_invoices(tmp_path)["abc12345"]
    delete_invoice(inv, tmp_path)
    assert not (tmp_path / "vectorstore" / "abc12345").exists()
    assert not (tmp_path / "data" / "abc12345").exists()


def test_delete_image_removes_file(tmp_path):
    img_dir = tmp_path / "data" / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "receipt.png").write_bytes(b"fake")
    inv = discover_invoices(tmp_path)["img_receipt.png"]
    delete_invoice(inv, tmp_path)
    assert not (img_dir / "receipt.png").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_store.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'store'`

- [ ] **Step 3: Implement store.py**

```python
# store.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_store.py -q`
Expected: 6 passed

- [ ] **Step 5: Wire into app.py**

In `app.py`, replace the session-state init:

```python
# OLD
if "invoices" not in st.session_state:
    st.session_state["invoices"] = {}
```

```python
# NEW
from store import discover_invoices, delete_invoice  # add to imports at top

if "invoices" not in st.session_state:
    st.session_state["invoices"] = discover_invoices(BASE_DIR)
```

And in the sidebar delete loop, replace:

```python
# OLD
for k in to_delete:
    del st.session_state["invoices"][k]
    st.rerun()
```

```python
# NEW
for k in to_delete:
    delete_invoice(st.session_state["invoices"][k], BASE_DIR)
    del st.session_state["invoices"][k]
    st.rerun()
```

- [ ] **Step 6: Run the fast suite (includes AppTest smoke tests)**

Run: `python -m pytest -m "not slow" -q`
Expected: all pass (39 fast tests after this task)

- [ ] **Step 7: Commit**

```bash
git add store.py tests/test_store.py app.py
git commit -m "feat: rediscover invoices on startup; delete removes files from disk"
```

---

### Task 3: OCR fallback for scanned PDFs

`pypdf.extract_text()` returns empty strings for scanned (image-only) PDFs — most real invoices. Fall back to rendering pages with pypdfium2 and reading them with RapidOCR (both pip-only, fully offline, models ship in the wheel).

**Files:**
- Create: `rag/ocr.py`
- Test: `tests/test_ocr.py`
- Modify: `ingest.py` (extract page texts via helper with fallback), `requirements.txt`

- [ ] **Step 1: Install the OCR dependencies (with this machine's onnxruntime pin)**

Run: `python -m pip install --no-cache-dir pypdfium2 rapidocr-onnxruntime "onnxruntime==1.19.2"`
Then verify: `python -c "import pypdfium2; from rapidocr_onnxruntime import RapidOCR; print('ocr deps OK')"`
Expected: `ocr deps OK`

- [ ] **Step 2: Add to requirements.txt**

Append these lines to `requirements.txt`:

```
pypdfium2>=4.0.0
rapidocr-onnxruntime>=1.3.0
onnxruntime>=1.17.0
```

- [ ] **Step 3: Write the failing tests**

```python
# tests/test_ocr.py
import pytest
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from ingest import _extract_page_texts


def _make_image_only_pdf(path: Path, text: str | None = None):
    img = Image.new("RGB", (1000, 700), "white")
    if text:
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(60)
        draw.text((80, 280), text, fill="black", font=font)
    img.save(path, "PDF")


def test_text_pdf_does_not_invoke_ocr(invoice_pdf, monkeypatch):
    import rag.ocr

    def _boom(*a, **k):
        raise AssertionError("OCR should not run for text PDFs")

    monkeypatch.setattr(rag.ocr, "ocr_pdf_pages", _boom)
    texts = _extract_page_texts(invoice_pdf)
    assert any(t.strip() for t in texts)


def test_scanned_pdf_falls_back_to_ocr(tmp_path, monkeypatch):
    import rag.ocr
    pdf = tmp_path / "scan.pdf"
    _make_image_only_pdf(pdf)

    monkeypatch.setattr(rag.ocr, "ocr_pdf_pages", lambda p: ["INVOICE TOTAL 212.09"])
    texts = _extract_page_texts(pdf)
    assert texts == ["INVOICE TOTAL 212.09"]


@pytest.mark.slow
def test_real_ocr_reads_rendered_text(tmp_path):
    from rag.ocr import ocr_pdf_pages
    pdf = tmp_path / "scan.pdf"
    _make_image_only_pdf(pdf, text="INVOICE 61356291")
    texts = ocr_pdf_pages(pdf)
    assert len(texts) == 1
    assert "61356291" in texts[0].replace(" ", "")
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/test_ocr.py -m "not slow" -q`
Expected: FAIL with `ImportError: cannot import name '_extract_page_texts' from 'ingest'`

- [ ] **Step 5: Implement rag/ocr.py**

```python
# rag/ocr.py
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
```

- [ ] **Step 6: Refactor ingest.py to use a page-text helper with OCR fallback**

In `ingest.py`, add this function above `ingest_pdf` (module level, lazy imports inside):

```python
_MIN_TEXT_CHARS = 32  # below this across all pages, treat the PDF as scanned


def _extract_page_texts(pdf_path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    page_texts = [(page.extract_text() or "") for page in reader.pages]
    if sum(len(t.strip()) for t in page_texts) < _MIN_TEXT_CHARS:
        import rag.ocr
        page_texts = rag.ocr.ocr_pdf_pages(pdf_path)
    return page_texts
```

Then inside `ingest_pdf`, replace the PDF-loading block:

```python
# OLD
    from pypdf import PdfReader
    # ... (keep the other two lazy imports)

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
```

```python
# NEW (PdfReader import moves into _extract_page_texts; keep the other lazy imports)
    # Load PDF and split per page (page numbers are 1-based for display)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE, chunk_overlap=cfg.CHUNK_OVERLAP
    )
    texts: list[str] = []
    pages: list[int] = []
    for page_num, page_text in enumerate(_extract_page_texts(pdf_path), start=1):
        for piece in splitter.split_text(page_text):
            texts.append(piece)
            pages.append(page_num)
```

Note: `_extract_page_texts` must do `import rag.ocr` + `rag.ocr.ocr_pdf_pages(...)` (module attribute access, exactly as written above) — a `from rag.ocr import ocr_pdf_pages` inside the function would bypass the tests' monkeypatching.

- [ ] **Step 7: Run fast OCR tests**

Run: `python -m pytest tests/test_ocr.py -m "not slow" -q`
Expected: 2 passed

- [ ] **Step 8: Run the real-OCR slow test**

Run: `python -m pytest tests/test_ocr.py -m slow -q`
Expected: 1 passed (first run initializes RapidOCR models, may take ~30s)

- [ ] **Step 9: Run full suite**

Run: `python -m pytest -q`
Expected: all pass

- [ ] **Step 10: Commit**

```bash
git add rag/ocr.py tests/test_ocr.py ingest.py requirements.txt
git commit -m "feat: OCR fallback for scanned PDFs via pypdfium2 + RapidOCR"
```

---

### Task 4: Extraction validation rules

The 3B local model makes arithmetic slips (we observed swapped qty/total in line items). Surface deterministic arithmetic checks in the Extract tab so users know when numbers don't add up.

**Files:**
- Create: `rag/validator.py`
- Test: `tests/test_validator.py`
- Modify: `app.py` (Extract tab)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_validator.py
from models.invoice import InvoiceSchema, LineItem
from rag.validator import validate_invoice


def test_consistent_invoice_has_no_warnings():
    schema = InvoiceSchema(
        subtotal=100.0, tax=10.0, total_amount=110.0,
        line_items=[LineItem(description="A", quantity=2, unit_price=25.0, total=50.0),
                    LineItem(description="B", quantity=1, unit_price=50.0, total=50.0)],
    )
    assert validate_invoice(schema) == []


def test_line_item_arithmetic_mismatch_flagged():
    schema = InvoiceSchema(
        line_items=[LineItem(description="Widget", quantity=2, unit_price=25.0, total=99.0)],
    )
    warnings = validate_invoice(schema)
    assert len(warnings) == 1
    assert "Widget" in warnings[0]


def test_subtotal_vs_items_mismatch_flagged():
    schema = InvoiceSchema(
        subtotal=500.0,
        line_items=[LineItem(description="A", total=50.0), LineItem(description="B", total=50.0)],
    )
    warnings = validate_invoice(schema)
    assert any("subtotal" in w.lower() for w in warnings)


def test_total_vs_subtotal_plus_tax_mismatch_flagged():
    schema = InvoiceSchema(subtotal=100.0, tax=10.0, total_amount=999.0)
    warnings = validate_invoice(schema)
    assert any("total" in w.lower() for w in warnings)


def test_negative_amount_flagged():
    schema = InvoiceSchema(total_amount=-5.0)
    warnings = validate_invoice(schema)
    assert any("negative" in w.lower() for w in warnings)


def test_missing_fields_produce_no_warnings():
    assert validate_invoice(InvoiceSchema()) == []


def test_small_rounding_differences_tolerated():
    schema = InvoiceSchema(subtotal=192.81, tax=19.28, total_amount=212.09)
    assert validate_invoice(schema) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_validator.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.validator'`

- [ ] **Step 3: Implement rag/validator.py**

```python
# rag/validator.py
from models.invoice import InvoiceSchema

_REL_TOLERANCE = 0.02   # 2% relative slack for OCR/LLM rounding
_ABS_TOLERANCE = 0.05   # at least 5 cents absolute


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(abs(b) * _REL_TOLERANCE, _ABS_TOLERANCE)


def validate_invoice(schema: InvoiceSchema) -> list[str]:
    """Deterministic arithmetic checks; returns human-readable warnings."""
    warnings: list[str] = []

    for i, li in enumerate(schema.line_items, 1):
        if li.quantity is not None and li.unit_price is not None and li.total is not None:
            expected = li.quantity * li.unit_price
            if not _close(expected, li.total):
                warnings.append(
                    f"Line {i} ({li.description[:40]}): qty × unit price = "
                    f"{expected:.2f}, but line total is {li.total:.2f}"
                )

    item_totals = [li.total for li in schema.line_items if li.total is not None]
    if item_totals and schema.subtotal is not None and not _close(sum(item_totals), schema.subtotal):
        warnings.append(
            f"Line items sum to {sum(item_totals):.2f}, but subtotal is {schema.subtotal:.2f}"
        )

    if schema.subtotal is not None and schema.tax is not None and schema.total_amount is not None:
        expected = schema.subtotal + schema.tax
        if not _close(expected, schema.total_amount):
            warnings.append(
                f"Subtotal + tax = {expected:.2f}, but total is {schema.total_amount:.2f}"
            )

    for field in ("subtotal", "tax", "total_amount"):
        value = getattr(schema, field)
        if value is not None and value < 0:
            warnings.append(f"{field} is negative: {value}")

    return warnings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_validator.py -q`
Expected: 7 passed

- [ ] **Step 5: Wire into the Extract tab in app.py**

Add `from rag.validator import validate_invoice` to the imports. Then in the Extract tab, directly after the metric-cards block (`m4.metric("Date", ...)`), insert:

```python
            checks = validate_invoice(cached)
            if checks:
                st.subheader("Validation checks")
                for w in checks:
                    st.warning(w)
            else:
                st.caption("All arithmetic checks pass.")
```

- [ ] **Step 6: Run the fast suite**

Run: `python -m pytest -m "not slow" -q`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add rag/validator.py tests/test_validator.py app.py
git commit -m "feat: arithmetic validation checks on extracted invoices"
```

---

### Task 5: Basic auth gate

Optional password gate: if `APP_PASSWORD` is set in the environment (or `.env`), the app demands it before rendering anything; unset means open (dev mode). Constant-time comparison via `hmac.compare_digest`.

**Files:**
- Create: `auth.py`
- Test: add to `tests/test_app.py`
- Modify: `app.py`

- [ ] **Step 1: Write the failing tests (append to tests/test_app.py)**

```python
def test_app_locked_when_password_set(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 0          # nothing past the gate renders
    assert len(at.text_input) == 1    # just the password prompt


def test_app_open_when_password_unset(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert len(at.tabs) == 3
```

- [ ] **Step 2: Run tests to verify the locked test fails**

Run: `python -m pytest tests/test_app.py -q`
Expected: `test_app_locked_when_password_set` FAILS (tabs render because no gate exists); the other tests pass

- [ ] **Step 3: Implement auth.py**

```python
# auth.py
import hmac
import os

import streamlit as st


def check_password() -> bool:
    """Gate the app behind APP_PASSWORD when set; open when unset (dev mode)."""
    expected = os.getenv("APP_PASSWORD", "")
    if not expected:
        return True
    if st.session_state.get("auth_ok"):
        return True

    def _verify():
        if hmac.compare_digest(st.session_state.get("password", ""), expected):
            st.session_state["auth_ok"] = True
            del st.session_state["password"]   # don't keep the secret around
        else:
            st.session_state["auth_ok"] = False

    st.text_input("Password", type="password", key="password", on_change=_verify)
    if st.session_state.get("auth_ok") is False:
        st.error("Incorrect password.")
    return False
```

- [ ] **Step 4: Wire into app.py**

Add `from auth import check_password` to the imports. Directly after the CSS injection line (`st.markdown(_CSS, unsafe_allow_html=True)`), insert:

```python
if not check_password():
    st.stop()
```

(Must come after `st.set_page_config` — which must stay the first Streamlit call — and before any session-state/sidebar work.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_app.py -q`
Expected: 4 passed

- [ ] **Step 6: Document in .env.example**

Replace the contents of `.env.example` with:

```
GOOGLE_API_KEY=your_gemini_api_key_here
# Optional: require a password to use the app (recommended when exposed on a network)
APP_PASSWORD=
```

- [ ] **Step 7: Run the fast suite, then commit**

Run: `python -m pytest -m "not slow" -q` — expected: all pass

```bash
git add auth.py tests/test_app.py app.py .env.example
git commit -m "feat: optional password gate via APP_PASSWORD"
```

---

### Task 6: Docker + compose with Ollama sidecar

Containerize the app and pair it with an Ollama service. Requires the Ollama URL to be configurable, so first extract an LLM factory that honors `OLLAMA_BASE_URL`.

**Files:**
- Create: `rag/llm.py`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- Test: `tests/test_llm.py`
- Modify: `app.py`, `rag/agent.py` (use the factory), `README.md` (run instructions)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm.py
from rag.llm import get_ollama_llm


def test_default_base_url(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    llm = get_ollama_llm("llama3.2:3b")
    assert llm.base_url == "http://localhost:11434"
    assert llm.model == "llama3.2:3b"
    assert llm.temperature == 0


def test_base_url_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama:11434")
    llm = get_ollama_llm("llama3.2:3b")
    assert llm.base_url == "http://ollama:11434"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_llm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.llm'`

- [ ] **Step 3: Implement rag/llm.py**

```python
# rag/llm.py
import os


def get_ollama_llm(model: str, temperature: float = 0):
    from langchain_ollama import OllamaLLM

    return OllamaLLM(
        model=model,
        temperature=temperature,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_llm.py -q`
Expected: 2 passed

- [ ] **Step 5: Use the factory everywhere**

In `app.py`, replace:

```python
# OLD
def _get_ollama_llm():
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model=cfg.LLM, temperature=0)
```

```python
# NEW
from rag.llm import get_ollama_llm   # add to imports at top

def _get_ollama_llm():
    return get_ollama_llm(cfg.LLM)
```

In `rag/agent.py`, replace:

```python
# OLD
    if llm is None:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(model=cfg.LLM, temperature=0)
```

```python
# NEW
    if llm is None:
        from rag.llm import get_ollama_llm
        llm = get_ollama_llm(cfg.LLM)
```

- [ ] **Step 6: Write the Docker files**

`Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

`docker-compose.yml`:

```yaml
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
      GOOGLE_API_KEY: ${GOOGLE_API_KEY:-}
      APP_PASSWORD: ${APP_PASSWORD:-}
    volumes:
      - app-data:/app/data
      - app-vectorstore:/app/vectorstore
    depends_on:
      - ollama

  ollama:
    image: ollama/ollama
    volumes:
      - ollama:/root/.ollama

volumes:
  app-data:
  app-vectorstore:
  ollama:
```

`.dockerignore`:

```
.git
.github
.pytest_cache
__pycache__
*.pyc
data/
vectorstore/
docs/
tests/
.env
*.png
```

- [ ] **Step 7: Add run instructions to README.md**

After the existing "### Run" section in `README.md`, add:

```markdown
### Run with Docker

```bash
docker compose up -d --build
docker compose exec ollama ollama pull llama3.2:3b   # first time only
```

Open `http://localhost:8501`. Set `GOOGLE_API_KEY` / `APP_PASSWORD` in your shell
or a `.env` file next to `docker-compose.yml` to pass them in.
```

- [ ] **Step 8: Build the image if Docker is available**

Run: `docker --version` — if Docker Desktop is installed and running:
Run: `docker build -t invoice-extractor .`
Expected: image builds successfully (the pip layer takes several minutes on first build).
If Docker is not available on this machine, validate the compose file instead: `python -c "import yaml; yaml.safe_load(open('docker-compose.yml')); print('valid')"` and note in the commit that the build was not executed locally.

- [ ] **Step 9: Run the full suite**

Run: `python -m pytest -q`
Expected: all pass

- [ ] **Step 10: Commit and push**

```bash
git add rag/llm.py tests/test_llm.py app.py rag/agent.py Dockerfile docker-compose.yml .dockerignore README.md
git commit -m "feat: Docker + compose with Ollama sidecar; configurable OLLAMA_BASE_URL"
git push origin master
```

- [ ] **Step 11: Confirm CI passes on GitHub**

Run: `gh run watch --repo Amanbatra03/invoice-extractor --exit-status`
Expected: CI green on the final push.

---

## Self-Review Notes

- Spec coverage: CI (Task 1), delete+rediscovery (Task 2), OCR (Task 3), validation (Task 4), auth (Task 5), Docker (Task 6) — all covered.
- `_extract_page_texts` is referenced in tests (Task 3 Step 3) and defined in Task 3 Step 6; monkeypatch requires module-attribute access (`rag.ocr.ocr_pdf_pages`), called out explicitly.
- `get_ollama_llm` signature consistent between Task 6 Steps 1/3/5.
- Auth gate placement constraint (after set_page_config) called out; existing `test_app_loads_without_exception` keeps passing because APP_PASSWORD is unset in the test env.
