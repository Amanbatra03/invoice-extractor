import asyncio
import os
import streamlit as st
from frontend.auth import is_authenticated, login_page, logout, get_token
from frontend.api_client import APIClient

st.set_page_config(page_title="Invoice Analyst", page_icon="🧾", layout="wide")

from frontend.theme import inject_theme
inject_theme()

if not is_authenticated():
    login_page()
    st.stop()

client = APIClient(base_url=os.getenv("API_BASE_URL", "http://localhost:8000"), token=get_token())

with st.sidebar:
    st.markdown(f"**{st.session_state.get('user_email', '')}**")
    if st.button("Sign Out"):
        logout()
    st.divider()
    uploaded = st.file_uploader("Upload Invoice (PDF or Image)", type=["pdf", "jpg", "jpeg", "png"])
    if uploaded and st.button("Add Invoice", type="primary"):
        with st.spinner("Uploading..."):
            try:
                result = asyncio.run(client.upload_invoice(
                    uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream"
                ))
                st.success(f"Uploaded — job {result['job_id']}")
            except Exception as e:
                st.error(str(e))

chat_tab, qa_tab, extract_tab, compare_tab, batch_tab, dashboard_tab = st.tabs(
    ["Chat", "Q&A", "Extract", "Compare", "Batch", "Dashboard"]
)

from frontend.pages import chat, qa, extract, compare, batch, dashboard
with chat_tab:
    chat.render(client)
with qa_tab:
    qa.render(client)
with extract_tab:
    extract.render(client)
with compare_tab:
    compare.render(client)
with batch_tab:
    batch.render(client)
with dashboard_tab:
    dashboard.render(client)
