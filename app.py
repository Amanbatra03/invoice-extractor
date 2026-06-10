import re
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


def _safe_filename(name: str, suffix: str) -> str:
    # Keep only the basename, then whitelist filename characters
    cleaned = re.sub(r"[^\w.\- ]", "_", Path(name).name).strip().strip(".")
    return cleaned or f"invoice{suffix}"


def _get_ollama_llm():
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model=cfg.LLM, temperature=0)


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
    cfg.NUM_RESULTS = st.number_input("NUM_RESULTS", min_value=1, max_value=10, value=int(cfg.NUM_RESULTS))
    cfg.MAX_AGENT_ITERATIONS = st.number_input("MAX_ITERS", min_value=1, max_value=5, value=int(cfg.MAX_AGENT_ITERATIONS))
    cfg.DEVICE = st.selectbox("Device", ["cpu", "cuda"], index=0)

# ── Tabs (always rendered) ────────────────────────────────────────────────────
qa_tab, extract_tab, compare_tab = st.tabs(["Q&A", "Extract", "Compare"])

# ── Invoice selector (inside main area, below tabs heading) ──────────────────
invoices = st.session_state["invoices"]

# ── Q&A Tab ───────────────────────────────────────────────────────────────────
with qa_tab:
    if not invoices:
        st.info("Upload an invoice in the sidebar to get started.")
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
        st.info("Upload an invoice in the sidebar to get started.")
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
