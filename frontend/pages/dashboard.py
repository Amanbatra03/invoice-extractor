import asyncio
import streamlit as st
from frontend.api_client import APIClient

_STATUS_ICON = {"done": "🟢", "failed": "🔴", "queued": "🟡", "running": "🔵"}


def render(client: APIClient):
    st.markdown(
        "### Dashboard"
        "<br><span style='color:#A8A599;font-size:0.85rem;'>"
        "Overview of your invoices and processing jobs.</span>",
        unsafe_allow_html=True,
    )
    st.write("")

    # ── Hero metrics ────────────────────────────────────────────
    try:
        data = asyncio.run(client.list_invoices(limit=200))
        items = data.get("items", [])
        total = data.get("total", 0)
        ready = sum(1 for i in items if i["status"] == "ready")
        pending = total - ready
    except Exception:
        items, total, ready, pending = [], 0, 0, 0

    try:
        jobs = asyncio.run(client.list_jobs(limit=50))
        failed_jobs = sum(1 for j in jobs if j["status"] == "failed")
    except Exception:
        jobs, failed_jobs = [], 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Invoices", total)
    m2.metric("Ready", ready)
    m3.metric("Processing", pending)
    m4.metric("Failed Jobs", failed_jobs, delta=None)

    st.write("")
    left, right = st.columns([3, 2], gap="large")

    # ── Invoice list ─────────────────────────────────────────────
    with left:
        st.markdown("**Invoices**")
        if not items:
            st.caption("No invoices yet — upload one from the sidebar.")
        else:
            for inv in items[:20]:
                icon = "🟢" if inv["status"] == "ready" else "🟡" if inv["status"] in ("pending", "ingesting") else "🔴"
                st.markdown(
                    f"{icon} &nbsp; `{inv['file_name']}` "
                    f"<span style='color:#A8A599;font-size:0.8rem;'>{inv['status']} · {inv['created_at'][:10]}</span>",
                    unsafe_allow_html=True,
                )
            if total > 20:
                st.caption(f"Showing 20 of {total} invoices.")

    # ── Recent jobs ──────────────────────────────────────────────
    with right:
        st.markdown("**Recent Jobs**")
        if not jobs:
            st.caption("No jobs yet.")
        else:
            for job in jobs[:10]:
                icon = _STATUS_ICON.get(job["status"], "⚪")
                st.markdown(
                    f"{icon} &nbsp; `{job['type']}` "
                    f"<span style='color:#A8A599;font-size:0.8rem;'>{job['status']} · {job['created_at'][:10]}</span>",
                    unsafe_allow_html=True,
                )
