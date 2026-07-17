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
    st.markdown(
        f'<div style="font-weight:600;font-size:0.95rem;color:#ECEAE4;margin-bottom:0.15rem;">'
        f'{st.session_state.get("user_email", "")}</div>'
        f'<div style="color:#A8A599;font-size:0.78rem;margin-bottom:1rem;">Signed in</div>',
        unsafe_allow_html=True,
    )
    if st.button("Sign Out", use_container_width=True):
        logout()
    st.divider()
    st.markdown("**Upload Invoice**")
    st.caption("PDF, JPG, or PNG — max 10 MB")
    uploaded = st.file_uploader("", type=["pdf", "jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded and st.button("Add Invoice", type="primary", use_container_width=True):
        with st.spinner("Uploading…"):
            try:
                result = asyncio.run(client.upload_invoice(
                    uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream"
                ))
                st.success(f"Queued — job `{result['job_id']}`")
            except Exception as e:
                st.error(str(e))

dashboard_tab, chat_tab, qa_tab, extract_tab, compare_tab, batch_tab = st.tabs(
    ["Dashboard", "Chat", "Q&A", "Extract", "Compare", "Batch"]
)

from frontend.pages import chat, qa, extract, compare, batch, dashboard
with dashboard_tab:
    dashboard.render(client)
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
