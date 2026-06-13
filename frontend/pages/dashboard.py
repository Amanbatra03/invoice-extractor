import asyncio
import streamlit as st
from frontend.api_client import APIClient


def render(client: APIClient):
    st.subheader("Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Recent Jobs")
        try:
            jobs = asyncio.run(client.list_jobs(limit=20))
            for job in jobs[:10]:
                status_color = "🟢" if job["status"] == "done" else "🔴" if job["status"] == "failed" else "🟡"
                st.markdown(f"{status_color} `{job['type']}` — {job['status']} — {job['created_at'][:10]}")
        except Exception as e:
            st.error(str(e))
    with col2:
        st.markdown("#### All Invoices")
        try:
            data = asyncio.run(client.list_invoices(limit=100))
            total = data.get("total", 0)
            items = data.get("items", [])
            ready = sum(1 for i in items if i["status"] == "ready")
            st.metric("Total", total)
            st.metric("Ready", ready)
            st.metric("Pending/Ingesting", total - ready)
        except Exception as e:
            st.error(str(e))
