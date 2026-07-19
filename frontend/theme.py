import streamlit as st

_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700'
    '&family=Space+Grotesk:wght@500;700&display=swap" rel="stylesheet">'
)

_THEME_CSS = """
<style>
:root {
    --bg:       #0A0A0A;
    --surface:  #111111;
    --border:   #2A2A2A;
    --accent:   #F5F500;
    --ink:      #F0F0F0;
    --ink-dim:  #666666;
    --red:      #FF3333;
    --green:    #00FF88;
    --blue:     #4488FF;
    --orange:   #FF8C00;
}

/* ── Dark background — all Streamlit root containers ── */
html, body,
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main, .main .block-container {
    background: #0A0A0A !important;
    background-color: #0A0A0A !important;
    color: #F0F0F0 !important;
}

/* ── Global font ──────────────────────────────────── */
html, body, *:not(code):not(pre) {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Hide default Streamlit chrome ────────────────── */
#MainMenu, footer, [data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
}

/* ── Hide Streamlit MPA auto-discovered nav ───────── */
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavLink"],
[data-testid="stSidebarNavItems"],
section[data-testid="stSidebar"] > div > ul {
    display: none !important;
}

/* ── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 2px solid #2A2A2A !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    min-width: 220px !important;
    max-width: 220px !important;
    padding: 0 !important;
}

[data-testid="stSidebarContent"] {
    padding: 1.5rem 0 1.5rem 0 !important;
}

/* ── Sidebar nav buttons ──────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-left: 3px solid transparent !important;
    border-radius: 0 !important;
    color: #666666 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-align: left !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.2rem !important;
    width: 100% !important;
    margin: 0 !important;
    box-shadow: none !important;
    transition: color 0.12s, border-color 0.12s, background 0.12s !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(245,245,0,0.06) !important;
    border-left-color: #F5F500 !important;
    color: #F0F0F0 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Nav active state ─────────────────────────────── */
[data-testid="stSidebar"] .nav-active-btn .stButton > button {
    background: rgba(245,245,0,0.10) !important;
    border-left: 3px solid #F5F500 !important;
    color: #F5F500 !important;
    font-weight: 700 !important;
}

/* ── Sidebar sign-out button ──────────────────────── */
[data-testid="stSidebar"] .signout-btn .stButton > button {
    background: transparent !important;
    border: 1px solid #FF3333 !important;
    border-radius: 0 !important;
    color: #FF3333 !important;
    font-size: 0.75rem !important;
    margin: 0 1rem !important;
    width: calc(100% - 2rem) !important;
    padding: 0.45rem 0.8rem !important;
    text-align: center !important;
}

[data-testid="stSidebar"] .signout-btn .stButton > button:hover {
    background: #FF3333 !important;
    color: #0A0A0A !important;
}

/* ── Headings ──────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Space Grotesk', 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    text-transform: uppercase !important;
    color: #F0F0F0 !important;
}

/* ── Metrics ──────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #111111 !important;
    border: 2px solid #2A2A2A !important;
    border-radius: 0 !important;
    padding: 1rem 1.2rem !important;
    box-shadow: none !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #666666 !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #F0F0F0 !important;
}

/* ── Main content buttons ──────────────────────────── */
[data-testid="stMain"] .stButton > button {
    background: transparent !important;
    border: 2px solid #2A2A2A !important;
    border-radius: 0 !important;
    color: #F0F0F0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1rem !important;
    transition: border-color 0.12s, background 0.12s, color 0.12s !important;
    box-shadow: none !important;
}

[data-testid="stMain"] .stButton > button:hover {
    background: #F5F500 !important;
    border-color: #F5F500 !important;
    color: #0A0A0A !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Primary button — Streamlit uses data-testid="baseButton-primary" */
[data-testid="baseButton-primary"],
[data-testid="stMain"] .stButton > button[kind="primary"],
[data-testid="stMain"] button[data-testid="baseButton-primary"] {
    background: #F5F500 !important;
    border-color: #F5F500 !important;
    color: #0A0A0A !important;
}

[data-testid="baseButton-primary"]:hover,
[data-testid="stMain"] .stButton > button[kind="primary"]:hover {
    background: #FFFFaa !important;
    border-color: #FFFFaa !important;
}

/* ── Inputs ────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #111111 !important;
    border: 2px solid #2A2A2A !important;
    border-radius: 0 !important;
    color: #F0F0F0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

[data-baseweb="select"] [data-baseweb="input"],
[data-baseweb="select"] > div {
    background: #111111 !important;
    border: 2px solid #2A2A2A !important;
    border-radius: 0 !important;
    color: #F0F0F0 !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #F5F500 !important;
    box-shadow: none !important;
    outline: none !important;
}

/* Placeholder text */
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: #444444 !important;
}

/* ── Selectbox / dropdown ──────────────────────────── */
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #111111 !important;
    border: 2px solid #2A2A2A !important;
    border-radius: 0 !important;
    color: #F0F0F0 !important;
}

/* ── File uploader ──────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #111111 !important;
    border: 2px dashed #2A2A2A !important;
    border-radius: 0 !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #F5F500 !important;
}

/* ── Dataframes / tables ────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 2px solid #2A2A2A !important;
    border-radius: 0 !important;
}

/* ── Expander ───────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 2px solid #2A2A2A !important;
    border-radius: 0 !important;
    background: #111111 !important;
}

[data-testid="stExpander"] details summary,
[data-testid="stExpanderDetails"] {
    background: #111111 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: #F0F0F0 !important;
}

/* ── Tabs (if any) ──────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #2A2A2A !important;
    gap: 0 !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 0 !important;
    color: #666666 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-bottom: 3px solid transparent !important;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: #F5F500 !important;
    border-bottom-color: #F5F500 !important;
    background: transparent !important;
}

/* ── Chat ───────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #111111 !important;
    border: 2px solid #2A2A2A !important;
    border-radius: 0 !important;
    padding: 0.85rem 1rem !important;
    margin-bottom: 0.5rem !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    border-left: 3px solid #F5F500 !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    border-left: 3px solid #4488FF !important;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] {
    background: #111111 !important;
    border: 2px solid #2A2A2A !important;
    border-radius: 0 !important;
    backdrop-filter: none !important;
}

/* ── Checkbox ───────────────────────────────────────── */
[data-testid="stCheckbox"] label {
    color: #F0F0F0 !important;
    font-size: 0.82rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Download button ────────────────────────────────── */
.stDownloadButton > button {
    background: transparent !important;
    border: 2px solid #00FF88 !important;
    border-radius: 0 !important;
    color: #00FF88 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

.stDownloadButton > button:hover {
    background: #00FF88 !important;
    color: #0A0A0A !important;
}

/* ── Progress bar ───────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background: #F5F500 !important;
    border-radius: 0 !important;
}

[data-testid="stProgressBar"] > div {
    background: #2A2A2A !important;
    border-radius: 0 !important;
}

/* ── Alerts / status ────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 0 !important;
    border-left-width: 4px !important;
    background: #111111 !important;
}

/* ── Spinner ───────────────────────────────────────── */
[data-testid="stSpinner"] {
    color: #F5F500 !important;
}

/* ── Dividers ───────────────────────────────────────── */
hr {
    border-color: #2A2A2A !important;
    border-width: 1px 0 0 0 !important;
    margin: 0.75rem 0 !important;
}

/* ── Scrollbar ──────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0A0A; }
::-webkit-scrollbar-thumb { background: #2A2A2A; }
::-webkit-scrollbar-thumb:hover { background: #666666; }

/* ── Custom components ──────────────────────────────── */
.brut-header {
    font-family: 'Space Grotesk', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #F0F0F0;
    border-bottom: 2px solid #F5F500;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

.brut-sub {
    font-size: 0.72rem;
    color: #666666;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.2rem;
    margin-bottom: 1rem;
}

.invoice-card {
    background: #111111;
    border: 2px solid #2A2A2A;
    padding: 0.85rem 1rem;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    transition: border-color 0.12s;
}

.invoice-card:hover {
    border-color: #F5F500;
}

.badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.18rem 0.5rem;
    border: 1px solid;
}

.badge-ready   { color: #00FF88; border-color: #00FF88; }
.badge-pending { color: #FF8C00; border-color: #FF8C00; }
.badge-failed  { color: #FF3333; border-color: #FF3333; }
.badge-running { color: #4488FF; border-color: #4488FF; }

/* ── Page entry animation ─── */
@media (prefers-reduced-motion: no-preference) {
    [data-testid="stMain"] > div > div {
        animation: pageIn 0.18s ease both;
    }
    [data-testid="stChatMessage"] {
        animation: msgIn 0.15s ease both;
    }
}

@keyframes pageIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: none; }
}

@keyframes msgIn {
    from { opacity: 0; transform: translateX(-6px); }
    to   { opacity: 1; transform: none; }
}

/* ── Typing dots ─────────────────────────────────────── */
.typing-dots { display: inline-flex; gap: 5px; padding: 4px 2px; }
.typing-dots span {
    width: 6px; height: 6px;
    background: #666666;
    display: inline-block;
    animation: dotBlink 0.9s step-start infinite;
}
.typing-dots span:nth-child(2) { animation-delay: 0.2s; }
.typing-dots span:nth-child(3) { animation-delay: 0.4s; }

@keyframes dotBlink {
    0%, 60%, 100% { opacity: 0.2; }
    30%            { opacity: 1; }
}
</style>
"""


def inject_theme() -> None:
    # Font via <link> (more reliable than @import in injected <style>)
    st.markdown(_FONT_LINK, unsafe_allow_html=True)
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
