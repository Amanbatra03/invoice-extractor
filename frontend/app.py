import os
import streamlit as st
from streamlit_cookies_controller import CookieController
from frontend.auth import is_authenticated, login_page, logout, get_token
from frontend.api_client import APIClient

st.set_page_config(page_title="Invoice Analyst", page_icon="▣", layout="wide")

from frontend.theme import inject_theme
inject_theme()

controller = CookieController()

# Restore session from cookie on refresh
if not st.session_state.get("access_token"):
    stored_token = controller.get("inv_token")
    stored_email = controller.get("inv_email")
    if stored_token:
        st.session_state["access_token"] = stored_token
        st.session_state["user_email"] = stored_email or ""

if not is_authenticated():
    login_page(controller)
    st.stop()

client = APIClient(
    base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
    token=get_token(),
)

# ── Navigation definition ──────────────────────────────────────────────────
NAV = [
    ("invoices", "▣", "INVOICES"),
    ("chat",     "◈", "CHAT"),
    ("qa",       "◎", "Q&A"),
    ("extract",  "◉", "EXTRACT"),
    ("compare",  "⊞", "COMPARE"),
    ("batch",    "◫", "BATCH"),
]

if "nav" not in st.session_state:
    st.session_state["nav"] = "invoices"

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:1rem 1.2rem 0.25rem;font-family:\'Space Grotesk\',monospace;'
        'font-size:1.1rem;font-weight:700;letter-spacing:0.06em;color:#F5F500;">'
        "INVOICE<br>ANALYST</div>"
        '<div style="padding:0 1.2rem 1rem;font-size:0.62rem;letter-spacing:0.12em;'
        'color:#666;text-transform:uppercase;">AI-powered · v1.0</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    current = st.session_state["nav"]
    for key, icon, label in NAV:
        active = current == key
        # Wrap in a div to allow CSS targeting via class
        div_cls = "nav-active-btn" if active else ""
        st.markdown(f'<div class="{div_cls}">', unsafe_allow_html=True)
        if st.button(f"{icon}  {label}", key=f"nav_{key}", use_container_width=True):
            st.session_state["nav"] = key
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Spacer + user info + sign out pinned to bottom
    st.markdown(
        '<div style="position:fixed;bottom:0;left:0;width:220px;'
        'background:#111;border-top:1px solid #2A2A2A;padding:0.8rem 1.2rem 1rem;">',
        unsafe_allow_html=True,
    )
    email = st.session_state.get("user_email", "")
    st.markdown(
        f'<div style="font-size:0.68rem;letter-spacing:0.06em;color:#666;'
        f'text-transform:uppercase;margin-bottom:0.4rem;word-break:break-all;">'
        f'{email}</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="signout-btn">', unsafe_allow_html=True)
    if st.button("Sign Out", key="signout", use_container_width=True):
        logout(controller)
    st.markdown("</div></div>", unsafe_allow_html=True)

# ── Page routing ───────────────────────────────────────────────────────────
from frontend.pages import invoices, chat, qa, extract, compare, batch

page = st.session_state["nav"]
{
    "invoices": invoices.render,
    "chat":     chat.render,
    "qa":       qa.render,
    "extract":  extract.render,
    "compare":  compare.render,
    "batch":    batch.render,
}[page](client)
