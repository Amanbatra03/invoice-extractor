import streamlit as st

_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

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

/* ── Reset & base ─────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] * {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg);
}

[data-testid="stMain"] {
    background: var(--bg);
}

/* ── Hide default Streamlit chrome ────────────────── */
#MainMenu, footer, [data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
}

/* ── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 2px solid var(--border) !important;
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
    color: var(--ink-dim) !important;
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
    border-left-color: var(--accent) !important;
    color: var(--ink) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Nav active state via data attribute ──────────── */
[data-testid="stSidebar"] .nav-active-btn .stButton > button {
    background: rgba(245,245,0,0.10) !important;
    border-left: 3px solid var(--accent) !important;
    color: var(--accent) !important;
    font-weight: 700 !important;
}

/* ── Sidebar sign-out button ──────────────────────── */
[data-testid="stSidebar"] .signout-btn .stButton > button {
    background: transparent !important;
    border: 1px solid var(--red) !important;
    border-radius: 0 !important;
    color: var(--red) !important;
    font-size: 0.75rem !important;
    margin: 0 1rem !important;
    width: calc(100% - 2rem) !important;
    padding: 0.45rem 0.8rem !important;
    text-align: center !important;
}

[data-testid="stSidebar"] .signout-btn .stButton > button:hover {
    background: var(--red) !important;
    color: #0A0A0A !important;
}

/* ── Headings ──────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Space Grotesk', 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    text-transform: uppercase !important;
}

/* ── Metrics ──────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 2px solid var(--border) !important;
    border-radius: 0 !important;
    padding: 1rem 1.2rem !important;
    box-shadow: none !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--ink-dim) !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--ink) !important;
}

/* ── Main content buttons ──────────────────────────── */
[data-testid="stMain"] .stButton > button {
    background: transparent !important;
    border: 2px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
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
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #0A0A0A !important;
    transform: none !important;
    box-shadow: none !important;
}

[data-testid="stMain"] .stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: #0A0A0A !important;
}

[data-testid="stMain"] .stButton > button[kind="primary"]:hover {
    background: #FFFFaa !important;
    border-color: #FFFFaa !important;
}

/* ── Inputs ────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"] div,
[data-testid="stSelectbox"] select {
    background: var(--surface) !important;
    border: 2px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: none !important;
    outline: none !important;
}

/* ── File uploader ──────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 0 !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}

/* ── Dataframes / tables ────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 2px solid var(--border) !important;
    border-radius: 0 !important;
}

/* ── Expander ───────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 2px solid var(--border) !important;
    border-radius: 0 !important;
    background: var(--surface) !important;
}

[data-testid="stExpander"] summary {
    background: var(--surface) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Chat ───────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 2px solid var(--border) !important;
    border-radius: 0 !important;
    padding: 0.85rem 1rem !important;
    margin-bottom: 0.5rem !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
}

[data-testid="stChatMessage"][data-testid*="user"] {
    border-left: 3px solid var(--accent) !important;
}

[data-testid="stChatMessage"][data-testid*="assistant"] {
    border-left: 3px solid var(--blue) !important;
}

[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 2px solid var(--border) !important;
    border-radius: 0 !important;
    backdrop-filter: none !important;
}

/* ── Download button ────────────────────────────────── */
.stDownloadButton > button {
    background: transparent !important;
    border: 2px solid var(--green) !important;
    border-radius: 0 !important;
    color: var(--green) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

.stDownloadButton > button:hover {
    background: var(--green) !important;
    color: #0A0A0A !important;
}

/* ── Progress bar ───────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background: var(--accent) !important;
    border-radius: 0 !important;
}

[data-testid="stProgressBar"] > div {
    background: var(--border) !important;
    border-radius: 0 !important;
}

/* ── Alerts / status ────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 0 !important;
    border-left-width: 4px !important;
}

/* ── Dividers ───────────────────────────────────────── */
hr {
    border-color: var(--border) !important;
    border-width: 1px 0 0 0 !important;
    margin: 0.75rem 0 !important;
}

/* ── Scrollbar ──────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); }
::-webkit-scrollbar-thumb:hover { background: var(--ink-dim); }

/* ── Custom components ──────────────────────────────── */
.brut-header {
    font-family: 'Space Grotesk', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--ink);
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

.brut-sub {
    font-size: 0.72rem;
    color: var(--ink-dim);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.2rem;
    margin-bottom: 1rem;
}

.invoice-card {
    background: var(--surface);
    border: 2px solid var(--border);
    padding: 0.85rem 1rem;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    transition: border-color 0.12s;
}

.invoice-card:hover {
    border-color: var(--accent);
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

.badge-ready   { color: var(--green);  border-color: var(--green);  }
.badge-pending { color: var(--orange); border-color: var(--orange); }
.badge-failed  { color: var(--red);    border-color: var(--red);    }
.badge-running { color: var(--blue);   border-color: var(--blue);   }

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
    background: var(--ink-dim);
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
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
