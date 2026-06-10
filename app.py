import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from auth import check_password
from ingest import ingest_pdf
from models.invoice import InvoiceSchema
from rag.agent import AgentState, build_agent
from rag.comparator import compare_invoices
from rag.extractor import extract_invoice
from rag.hybrid_retriever import HybridRetriever
from rag.utils import load_config
from rag.validator import validate_invoice
from store import discover_invoices, delete_invoice
from vision.gemini import ask_invoice, extract_invoice_gemini

load_dotenv()
cfg = load_config()

BASE_DIR = Path(__file__).parent

st.set_page_config(page_title="Invoice Analyst", page_icon="🧾", layout="wide")

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Outfit:wght@400;500;600&display=swap');

html, body,
[data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]),
[data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]) {
    font-family: 'Outfit', 'Segoe UI', sans-serif;
}
h1, h2, h3, [data-testid="stMetricValue"] {
    font-family: 'Fraunces', Georgia, serif !important;
    letter-spacing: -0.015em;
}

/* Brand header */
.stMarkdown .brand-title {
    font-family: 'Fraunces', Georgia, serif !important;
    font-size: 2.1rem !important;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin: 0;
    color: #ECEAE4;
}
.stMarkdown .brand-tagline {
    color: #A8A599;
    font-size: 0.92rem;
    margin: 0.15rem 0 0 0;
}
.brand-rule {
    border: none;
    border-top: 1px solid #3E3D39;
    margin: 0.9rem 0 0.4rem 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1F1E1B;
    border-right: 1px solid #3E3D39;
}
.step-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #D97757;
    margin-bottom: 0;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.01em;
    transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(217, 119, 87, 0.25);
}
.stButton > button:active, .stDownloadButton > button:active {
    transform: scale(0.98);
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 2px solid #D97757;
    outline-offset: 2px;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
    letter-spacing: 0.02em;
    font-size: 0.95rem;
}

/* Invoice list rows */
.inv-row { padding: 0.1rem 0; }
.inv-badge {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 0.1rem 0.4rem;
    margin-left: 0.45rem;
    border-radius: 4px;
    background: #42312A;
    color: #E0916F;
    vertical-align: middle;
}
.inv-badge.image { background: #34332F; color: #B5B2A6; }

/* Empty state */
.empty-state {
    background: #30302E;
    border-radius: 14px;
    padding: 2.2rem 2.4rem;
    margin-top: 0.8rem;
}
.empty-state h3 {
    margin: 0 0 0.5rem 0;
    font-size: 1.25rem;
}
.empty-state p {
    color: #A8A599;
    margin: 0.2rem 0;
    max-width: 38rem;
}
.empty-state .kbd {
    background: #3E3D39;
    border-radius: 5px;
    padding: 0.05rem 0.45rem;
    font-size: 0.85em;
    color: #ECEAE4;
}

/* Sidebar expander — keep it dark like the rest of the sidebar */
[data-testid="stSidebar"] [data-testid="stExpander"] details {
    background: #2A2927;
    border-color: #3E3D39;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #ECEAE4;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #30302E;
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
}
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }

/* Dataframes / numbers */
[data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

if not check_password():
    st.stop()

# Session state — rediscover previously ingested invoices from disk
if "invoices" not in st.session_state:
    st.session_state["invoices"] = discover_invoices(BASE_DIR)


def _safe_filename(name: str, suffix: str) -> str:
    # Keep only the basename, then whitelist filename characters
    cleaned = re.sub(r"[^\w.\- ]", "_", Path(name).name).strip().strip(".")
    return cleaned or f"invoice{suffix}"


def _get_ollama_llm():
    from rag.llm import get_ollama_llm
    return get_ollama_llm(cfg.LLM)


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


def _empty_state():
    st.markdown(
        """
        <div class="empty-state">
          <h3>No invoices loaded yet</h3>
          <p>Use the sidebar to get started — choose a PDF or image,
          then press <span class="kbd">Add Invoice</span>.</p>
          <p>PDFs are analysed fully offline with a local model.
          Images are read by Gemini and need an API key in <span class="kbd">.env</span>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload")

    st.markdown('<p class="step-label">STEP 1 · CHOOSE A FILE</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("PDF or Image", type=["pdf", "jpg", "jpeg", "png"])

    st.markdown('<p class="step-label">STEP 2 · LOAD IT</p>', unsafe_allow_html=True)
    add_clicked = st.button(
        "Add Invoice", type="primary", use_container_width=True,
        disabled=uploaded is None,
    )
    if uploaded is None:
        st.caption("The button unlocks once a file is chosen.")

    if uploaded and add_clicked:
        suffix = Path(uploaded.name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        if suffix == ".pdf":
            with st.spinner(f"Ingesting {uploaded.name}…"):
                try:
                    sha_key = ingest_pdf(
                        tmp_path, base_dir=BASE_DIR,
                        original_name=_safe_filename(uploaded.name, suffix),
                    )
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
            safe_name = _safe_filename(uploaded.name, suffix)
            dest = img_dest / safe_name
            dest.write_bytes(uploaded.getvalue())
            img_key = f"img_{safe_name}"
            st.session_state["invoices"][img_key] = {
                "name": safe_name,
                "type": "image",
                "path": dest,
                "schema_cache": None,
            }
            st.success(f"Loaded: {safe_name}")

        tmp_path.unlink(missing_ok=True)

    st.divider()
    st.subheader("Loaded invoices")
    if not st.session_state["invoices"]:
        st.caption("Nothing loaded yet.")
    to_delete = []
    for key, inv in st.session_state["invoices"].items():
        col1, col2 = st.columns([4, 1])
        badge_cls = "inv-badge image" if inv["type"] == "image" else "inv-badge"
        col1.markdown(
            f'<div class="inv-row"><strong>{inv["name"]}</strong>'
            f'<span class="{badge_cls}">{inv["type"]}</span></div>',
            unsafe_allow_html=True,
        )
        if col2.button("✕", key=f"del_{key}", help=f"Remove {inv['name']}"):
            to_delete.append(key)
    for k in to_delete:
        delete_invoice(st.session_state["invoices"][k], BASE_DIR)
        del st.session_state["invoices"][k]
        st.rerun()

    st.divider()
    with st.expander("Retrieval settings"):
        cfg.NUM_RESULTS = st.number_input("Chunks per query", min_value=1, max_value=10, value=int(cfg.NUM_RESULTS))
        cfg.MAX_AGENT_ITERATIONS = st.number_input("Max query retries", min_value=1, max_value=5, value=int(cfg.MAX_AGENT_ITERATIONS))
        cfg.DEVICE = st.selectbox("Device", ["cpu", "cuda"], index=0)

# ── Brand header ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <p class="brand-title">Invoice Analyst</p>
    <p class="brand-tagline">offline RAG for PDFs · Gemini vision for images · structured extraction & comparison</p>
    <hr class="brand-rule" />
    """,
    unsafe_allow_html=True,
)

# ── Tabs (always rendered) ────────────────────────────────────────────────────
qa_tab, extract_tab, compare_tab = st.tabs(["Q&A", "Extract", "Compare"])

invoices = st.session_state["invoices"]

# ── Q&A Tab ───────────────────────────────────────────────────────────────────
with qa_tab:
    if not invoices:
        _empty_state()
    else:
        invoice_names = {k: v["name"] for k, v in invoices.items()}
        selected_key = st.selectbox(
            "Active invoice", list(invoice_names.keys()), format_func=lambda k: invoice_names[k], key="qa_select"
        )
        selected = invoices[selected_key]

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
                except (EnvironmentError, ValueError) as e:
                    st.error(str(e))
                except RuntimeError as e:
                    st.warning(str(e))

# ── Extract Tab ───────────────────────────────────────────────────────────────
with extract_tab:
    if not invoices:
        _empty_state()
    else:
        invoice_names = {k: v["name"] for k, v in invoices.items()}
        selected_key_ext = st.selectbox(
            "Active invoice", list(invoice_names.keys()), format_func=lambda k: invoice_names[k], key="ext_select"
        )
        selected_ext = invoices[selected_key_ext]

        if st.button("Extract All Fields", type="primary", key="extract_btn"):
            if selected_ext["type"] == "pdf":
                try:
                    retriever = HybridRetriever(selected_ext["sha_key"], base_dir=BASE_DIR)
                    llm = _get_ollama_llm()
                    with st.spinner("Extracting structured fields…"):
                        schema = extract_invoice(retriever, llm)
                    invoices[selected_key_ext]["schema_cache"] = schema
                except Exception as e:
                    st.error(f"Extraction failed: {e}\n\nMake sure Ollama is running: `ollama serve`")
            else:
                try:
                    with st.spinner("Extracting via Gemini…"):
                        schema = extract_invoice_gemini(selected_ext["path"])
                    invoices[selected_key_ext]["schema_cache"] = schema
                except (EnvironmentError, ValueError) as e:
                    st.error(str(e))

        cached = invoices[selected_key_ext].get("schema_cache")
        if cached:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total", f"{cached.total_amount:,.2f}" if cached.total_amount is not None else "—")
            m2.metric("Vendor", cached.vendor_name or "—")
            m3.metric("Invoice #", cached.invoice_number or "—")
            m4.metric("Date", cached.invoice_date or "—")

            checks = validate_invoice(cached)
            if checks:
                st.subheader("Validation checks")
                for w in checks:
                    st.warning(w)
            else:
                st.caption("All arithmetic checks pass.")

            header_df, items_df = _schema_to_dfs(cached)
            st.subheader("Header fields")
            st.dataframe(header_df, use_container_width=True, hide_index=True)
            if not items_df.empty:
                st.subheader("Line items")
                st.dataframe(items_df, use_container_width=True, hide_index=True)
            col1, col2 = st.columns(2)
            col1.download_button(
                "Download JSON",
                data=cached.model_dump_json(indent=2),
                file_name=f"{selected_ext['name']}_extracted.json",
                mime="application/json",
            )
            col2.download_button(
                "Download CSV",
                data=header_df.to_csv(index=False),
                file_name=f"{selected_ext['name']}_extracted.csv",
                mime="text/csv",
            )

# ── Compare Tab ───────────────────────────────────────────────────────────────
with compare_tab:
    pdf_invoices = {k: v for k, v in invoices.items() if v["type"] == "pdf"}
    if len(pdf_invoices) < 2:
        st.markdown(
            """
            <div class="empty-state">
              <h3>Comparison needs two PDFs</h3>
              <p>Load at least two PDF invoices in the sidebar, then pick the ones
              to put side by side. Discrepancies in vendor, totals and dates are
              flagged automatically.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
                st.subheader("Side-by-side comparison")
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
                            styles.loc[field] = "background-color: #5C3A2B"
                    return styles

                st.dataframe(compare_df.style.apply(highlight_discrepancies, axis=None), use_container_width=True)

            if discrepancies:
                st.subheader("Discrepancies")
                for d in discrepancies:
                    st.warning(f"**{d['field']}**: {d['detail']}")
            else:
                st.success("No discrepancies found.")
