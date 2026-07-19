import asyncio
import streamlit as st
from frontend.api_client import APIClient


def render(client: APIClient):
    st.markdown('<div class="brut-header">Q&A</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brut-sub">SINGLE QUESTION · ONE INVOICE</div>',
        unsafe_allow_html=True,
    )

    try:
        invoices_data = asyncio.run(client.list_invoices(limit=100))
        invoices = invoices_data.get("items", [])
    except Exception as e:
        st.error(str(e))
        return

    if not invoices:
        st.markdown(
            '<div style="border:2px solid #2A2A2A;padding:1.5rem;color:#666;'
            'font-size:0.78rem;letter-spacing:0.05em;">NO INVOICES FOUND. '
            '<a href="#" style="color:#F5F500;">GO TO INVOICES PAGE TO UPLOAD.</a></div>',
            unsafe_allow_html=True,
        )
        return

    ready_invoices = [inv for inv in invoices if inv.get("status") == "ready"]
    if not ready_invoices:
        st.markdown(
            '<div style="border:2px solid #2A2A2A;padding:1.5rem;color:#666;font-size:0.78rem;">'
            'NO READY INVOICES. WAIT FOR INGESTION TO COMPLETE OR UPLOAD FROM THE INVOICES PAGE.</div>',
            unsafe_allow_html=True,
        )
        return

    options = {inv["id"]: inv["file_name"] for inv in ready_invoices}

    # Pre-select if navigated from Invoices page
    default_idx = 0
    preselect = st.session_state.pop("preselect_invoice_id", None)
    if preselect and preselect in options:
        default_idx = list(options.keys()).index(preselect)

    selected_id = st.selectbox(
        "INVOICE",
        list(options.keys()),
        index=default_idx,
        format_func=lambda k: options[k],
        key="qa_invoice_sel",
    )

    # Clear cached result if user switches invoices
    if st.session_state.get("qa_result_invoice") != selected_id:
        st.session_state.pop("qa_result", None)
        st.session_state.pop("qa_error", None)

    question = st.text_input(
        "QUESTION",
        placeholder="WHAT IS THE INVOICE TOTAL?",
        key="qa_question_input",
    )

    if st.button("ASK", type="primary") and question.strip():
        with st.spinner("RUNNING..."):
            try:
                result = asyncio.run(client.ask_question(selected_id, question))
                st.session_state["qa_result"] = result
                st.session_state["qa_result_invoice"] = selected_id
                st.session_state["qa_result_question"] = question
                st.session_state.pop("qa_error", None)
            except Exception as e:
                st.session_state["qa_error"] = str(e)
                st.session_state.pop("qa_result", None)

    # Persist answer across reruns
    if "qa_result" in st.session_state and st.session_state.get("qa_result_invoice") == selected_id:
        result = st.session_state["qa_result"]
        saved_q = st.session_state.get("qa_result_question", "")
        answer = result.get("answer", "No answer generated.")

        st.markdown(
            f'<div style="font-size:0.65rem;letter-spacing:0.1em;color:#666;margin-top:1.2rem;margin-bottom:0.3rem;">'
            f'Q: {saved_q.upper()}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="border:2px solid #F5F500;padding:1.2rem;font-size:0.85rem;'
            f'color:#F0F0F0;background:#111;line-height:1.6;">{answer}</div>',
            unsafe_allow_html=True,
        )

        if result.get("agent_trace"):
            with st.expander("AGENT TRACE"):
                for step in result["agent_trace"]:
                    st.json(step)

        if result.get("chunks"):
            with st.expander(f"SOURCE CHUNKS ({len(result['chunks'])})"):
                for i, chunk in enumerate(result["chunks"], 1):
                    st.markdown(
                        f'<div style="font-size:0.72rem;color:#F5F500;letter-spacing:0.06em;'
                        f'margin-bottom:0.2rem;">CHUNK {i} · PAGE {chunk.get("page", "?")} · '
                        f'SCORE {chunk.get("score", 0):.4f}</div>',
                        unsafe_allow_html=True,
                    )
                    st.code(chunk.get("text", "")[:400], language=None)

    elif "qa_error" in st.session_state:
        st.error(st.session_state["qa_error"])
