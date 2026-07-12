import streamlit as st

_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Outfit:wght@400;500;600&display=swap');

:root {
    --glass-bg: rgba(44, 43, 40, 0.55);
    --glass-bg-strong: rgba(38, 37, 34, 0.72);
    --glass-border: rgba(255, 255, 255, 0.09);
    --glass-highlight: rgba(255, 255, 255, 0.22);
    --accent: #D97757;
    --ink: #ECEAE4;
    --ink-dim: #A8A599;
}

/* Typography */
html, body,
[data-testid="stAppViewContainer"] *:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]),
[data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]):not([class*="material-symbols"]) {
    font-family: 'Outfit', 'Segoe UI', sans-serif;
    color: var(--ink);
}
h1, h2, h3, [data-testid="stMetricValue"] {
    font-family: 'Fraunces', Georgia, serif !important;
    letter-spacing: -0.015em;
}

/* Animated depth backdrop — gives the glass something to refract */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(60% 80% at 15% 10%, rgba(217, 119, 87, 0.14), transparent 60%),
        radial-gradient(50% 70% at 85% 85%, rgba(120, 140, 200, 0.10), transparent 60%),
        radial-gradient(40% 55% at 70% 20%, rgba(217, 170, 87, 0.07), transparent 55%),
        #1D1C19;
    background-size: 160% 160%;
}

/* Glass surfaces */
[data-testid="stSidebar"],
[data-testid="stChatMessage"],
[data-testid="stMetric"],
[data-testid="stExpander"] details,
.glass-panel {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    border: 1px solid var(--glass-border);
    border-top-color: var(--glass-highlight);
    border-radius: 18px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}
[data-testid="stSidebar"] {
    border-radius: 0;
    border-top-color: var(--glass-border);
    border-right: 1px solid var(--glass-border);
}
[data-testid="stChatMessage"] {
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.35rem;
}

/* Chat input pinned bar */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div {
    background: var(--glass-bg-strong) !important;
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    border-radius: 16px;
    border: 1px solid var(--glass-border);
}

/* Buttons — lift, sheen, spring */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px;
    font-weight: 600;
    letter-spacing: 0.01em;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--glass-border);
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015));
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(217, 119, 87, 0.28);
    border-color: rgba(217, 119, 87, 0.45);
}
.stButton > button:active, .stDownloadButton > button:active {
    transform: scale(0.97);
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
    letter-spacing: 0.02em;
    font-size: 0.95rem;
}

/* Metric numbers */
[data-testid="stMetric"] { padding: 0.9rem 1.1rem; }
[data-testid="stMetricValue"], [data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; }

/* Typing indicator */
.typing-dots { display: inline-flex; gap: 6px; padding: 6px 2px; }
.typing-dots span {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--ink-dim);
    opacity: 0.3;
}

/* Empty state panel */
.glass-empty {
    background: var(--glass-bg);
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    border: 1px solid var(--glass-border);
    border-top-color: var(--glass-highlight);
    border-radius: 18px;
    padding: 2.2rem 2.4rem;
    margin-top: 0.8rem;
}
.glass-empty h3 { margin: 0 0 0.5rem 0; }
.glass-empty p { color: var(--ink-dim); margin: 0.2rem 0; max-width: 38rem; }

/* Motion — only for users who haven't opted out */
@media (prefers-reduced-motion: no-preference) {
    [data-testid="stAppViewContainer"] {
        animation: glassDrift 26s ease-in-out infinite alternate;
    }
    [data-testid="stChatMessage"] {
        animation: msgIn 0.45s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .stButton > button, .stDownloadButton > button {
        transition: transform 0.2s cubic-bezier(0.22, 1, 0.36, 1),
                    box-shadow 0.2s ease, border-color 0.2s ease;
    }
    button[data-baseweb="tab"] { transition: color 0.18s ease; }
    .typing-dots span { animation: dotPulse 1.2s ease-in-out infinite; }
    .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.3s; }
}

@keyframes glassDrift {
    from { background-position: 0% 0%; }
    to { background-position: 100% 100%; }
}
@keyframes msgIn {
    from { opacity: 0; transform: translateY(14px) scale(0.98); }
    to { opacity: 1; transform: none; }
}
@keyframes dotPulse {
    0%, 60%, 100% { opacity: 0.25; transform: translateY(0); }
    30% { opacity: 1; transform: translateY(-4px); }
}
</style>
"""


def inject_theme() -> None:
    """Apply the liquid-glass theme. Call once per page render, right after set_page_config."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
