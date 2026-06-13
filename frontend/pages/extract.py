import asyncio
import pandas as pd
import streamlit as st
from frontend.api_client import APIClient
from models.invoice import InvoiceSchema


def render(client: APIClient):
    st.subheader("Extract")
    try:
        invoices_data = asyncio.run(client.list_invoices())
        invoices = invoices_data.get("items", [])
    except Exception as e:
        st.error(str(e))
        return
    if not invoices:
        st.info("No invoices loaded yet.")
        return
    options = {inv["id"]: inv["file_name"] for inv in invoices}
    selected_id = st.selectbox("Invoice", list(options.keys()), format_func=lambda k: options[k], key="ext_sel")
    if st.button("Extract All Fields", type="primary"):
        with st.spinner("Extracting..."):
            try:
                job = asyncio.run(client.run_extraction(selected_id))
                st.info(f"Extraction queued — job {job['job_id']}. Refresh to see results.")
            except Exception as e:
                st.error(str(e))
    try:
        ext = asyncio.run(client.get_extraction(selected_id))
        schema = InvoiceSchema.model_validate(ext["schema_json"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", f"{schema.total_amount:,.2f}" if schema.total_amount else "—")
        c2.metric("Vendor", schema.vendor_name or "—")
        c3.metric("Invoice #", schema.invoice_number or "—")
        c4.metric("Date", schema.invoice_date or "—")
        st.subheader("Header fields")
        st.dataframe(pd.DataFrame({
            "Field": ["Vendor", "Invoice #", "Date", "Due Date", "Subtotal", "Tax", "Total", "Currency"],
            "Value": [schema.vendor_name, schema.invoice_number, schema.invoice_date,
                      schema.due_date, schema.subtotal, schema.tax, schema.total_amount, schema.currency],
        }), use_container_width=True, hide_index=True)
        if schema.line_items:
            st.subheader("Line items")
            st.dataframe(pd.DataFrame([li.model_dump() for li in schema.line_items]), use_container_width=True)
        st.download_button("Download JSON", schema.model_dump_json(indent=2), "extraction.json", "application/json")
    except Exception:
        st.caption("No extraction yet — click Extract above.")
