import asyncio
import pandas as pd
import streamlit as st
from frontend.api_client import APIClient
from models.invoice import InvoiceSchema

_STATUS_COLOR = {
    "ready":     "#00FF88",
    "failed":    "#FF3333",
    "pending":   "#FF8C00",
    "ingesting": "#FF8C00",
}


def render(client: APIClient):
    st.markdown('<div class="brut-header">EXTRACT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brut-sub">STRUCTURED FIELDS · VENDOR · DATES · TOTALS · LINE ITEMS</div>',
        unsafe_allow_html=True,
    )

    try:
        invoices_data = asyncio.run(client.list_invoices(limit=100))
        invoices = invoices_data.get("items", [])
    except Exception as e:
        st.error(str(e))
        return

    if not invoices:
        st.markdown(
            '<div style="border:2px solid #2A2A2A;padding:1.5rem;color:#666;font-size:0.78rem;">'
            'NO INVOICES FOUND. UPLOAD FROM THE INVOICES PAGE.</div>',
            unsafe_allow_html=True,
        )
        return

    options = {inv["id"]: inv["file_name"] for inv in invoices}

    # Pre-select if navigated from Invoices page
    default_idx = 0
    preselect = st.session_state.pop("preselect_invoice_id", None)
    if preselect and preselect in options:
        default_idx = list(options.keys()).index(preselect)

    sel_col, status_col = st.columns([4, 1])
    with sel_col:
        selected_id = st.selectbox(
            "INVOICE",
            list(options.keys()),
            index=default_idx,
            format_func=lambda k: options[k],
            key="ext_sel",
        )

    # Find selected invoice's status
    selected_inv = next((inv for inv in invoices if inv["id"] == selected_id), None)
    inv_status = selected_inv.get("status", "") if selected_inv else ""
    status_color = _STATUS_COLOR.get(inv_status, "#666")

    with status_col:
        st.markdown(
            f'<div style="margin-top:1.85rem;">'
            f'<span class="badge" style="color:{status_color};border-color:{status_color};">'
            f'{inv_status.upper()}</span></div>',
            unsafe_allow_html=True,
        )

    can_extract = inv_status in ("ready", "failed")

    if not can_extract:
        st.markdown(
            f'<div style="border:2px solid #FF8C00;padding:0.75rem 1rem;font-size:0.78rem;'
            f'color:#FF8C00;margin:0.5rem 0;">INVOICE IS {inv_status.upper()} — '
            f'WAIT FOR INGESTION TO COMPLETE BEFORE EXTRACTING.</div>',
            unsafe_allow_html=True,
        )

    extract_col, _ = st.columns([2, 5])
    with extract_col:
        if st.button("EXTRACT FIELDS", type="primary", disabled=not can_extract):
            with st.spinner("EXTRACTING..."):
                try:
                    job = asyncio.run(client.run_extraction(selected_id))
                    st.session_state["ext_job_id"] = job.get("job_id", "")
                    st.info(f"QUEUED — JOB {job.get('job_id', '')}. RESULTS APPEAR BELOW WHEN READY.")
                except Exception as e:
                    st.error(str(e))

    st.divider()

    # Try to load existing extraction
    try:
        ext = asyncio.run(client.get_extraction(selected_id))
        schema = InvoiceSchema.model_validate(ext["schema_json"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TOTAL", f"{schema.total_amount:,.2f}" if schema.total_amount else "—")
        c2.metric("VENDOR", schema.vendor_name or "—")
        c3.metric("INVOICE #", schema.invoice_number or "—")
        c4.metric("DATE", schema.invoice_date or "—")

        st.markdown(
            '<div class="brut-sub" style="margin-top:1.5rem;">HEADER FIELDS</div>',
            unsafe_allow_html=True,
        )
        fields = [
            ("VENDOR", schema.vendor_name),
            ("INVOICE #", schema.invoice_number),
            ("DATE", schema.invoice_date),
            ("DUE DATE", schema.due_date),
            ("PO NUMBER", schema.po_number),
            ("PAYMENT TERMS", schema.payment_terms),
            ("SUBTOTAL", schema.subtotal),
            ("TAX", schema.tax),
            ("TOTAL", schema.total_amount),
            ("CURRENCY", schema.currency),
            ("VENDOR TAX ID", schema.vendor_tax_id),
            ("VENDOR ADDRESS", schema.vendor_address),
            ("BILL TO", schema.bill_to),
        ]
        df = pd.DataFrame(
            [(f, str(v) if v is not None else "—") for f, v in fields],
            columns=["FIELD", "VALUE"],
        )
        st.dataframe(df, use_container_width=True, hide_index=True)

        if schema.line_items:
            st.markdown(
                f'<div class="brut-sub" style="margin-top:1.5rem;">LINE ITEMS ({len(schema.line_items)})</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                pd.DataFrame([li.model_dump() for li in schema.line_items]),
                use_container_width=True,
            )

        dl_col, _ = st.columns([2, 5])
        with dl_col:
            st.download_button(
                "DOWNLOAD JSON",
                schema.model_dump_json(indent=2),
                "extraction.json",
                "application/json",
                use_container_width=True,
            )

    except Exception:
        st.markdown(
            '<div style="color:#444;font-size:0.82rem;letter-spacing:0.04em;padding:0.5rem 0;">'
            'NO EXTRACTION YET — CLICK EXTRACT FIELDS ABOVE.</div>',
            unsafe_allow_html=True,
        )
