import asyncio
import streamlit as st
from frontend.api_client import APIClient


def render(client: APIClient):
    st.subheader("Q&A")
    try:
        invoices_data = asyncio.run(client.list_invoices())
        invoices = invoices_data.get("items", [])
    except Exception as e:
        st.error(str(e))
        return
    if not invoices:
        st.info("No invoices loaded yet — upload one from the sidebar.")
        return
    options = {inv["id"]: inv["file_name"] for inv in invoices}
    selected_id = st.selectbox("Invoice", list(options.keys()), format_func=lambda k: options[k])
    question = st.text_input("Ask a question", placeholder="e.g. What is the invoice total?")
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Running agentic RAG..."):
            try:
                result = asyncio.run(client.ask_question(selected_id, question))
                st.success(result.get("answer", "No answer generated."))
                with st.expander("Agent trace"):
                    for step in result.get("agent_trace", []):
                        st.json(step)
                if result.get("chunks"):
                    with st.expander("Source chunks"):
                        for i, chunk in enumerate(result["chunks"], 1):
                            st.markdown(f"**Chunk {i}** — page `{chunk.get('page')}`, score `{chunk.get('score', 0):.4f}`")
                            st.text(chunk.get("text", "")[:400])
            except Exception as e:
                st.error(str(e))
