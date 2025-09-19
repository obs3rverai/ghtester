# ui/tabs/verify_tab.py
from __future__ import annotations

import io
import os
from pathlib import Path
import requests
import streamlit as st

def _get_api_base() -> str:
    # secrets → env → default
    try:
        return st.secrets["API_BASE"]
    except Exception:
        pass
    envv = os.environ.get("API_BASE")
    if envv:
        return envv
    return "http://127.0.0.1:8000/api"

API_BASE = _get_api_base()

def _post_verify(pdf_bytes: bytes, pub_pem: bytes | None, bundle_bytes: bytes | None):
    files = {"pdf": ("report.pdf", pdf_bytes, "application/pdf")}
    if pub_pem:
        files["public_key_pem"] = ("public.pem", pub_pem, "application/x-pem-file")
    if bundle_bytes:
        files["bundle_json"] = ("bundle.json", bundle_bytes, "application/json")
    r = requests.post(f"{API_BASE}/verify/pdf", files=files, timeout=120)
    r.raise_for_status()
    return r.json()

def render():
    st.header("Verify Report (PDF)")
    st.caption("Upload a generated PDF. Optionally add a public key (PEM) and/or the report’s JSON bundle for cross-check.")
    c1, c2 = st.columns([2, 1])
    with c1:
        up_pdf = st.file_uploader("Report PDF", type=["pdf"], key="ver_pdf")
        up_pem = st.file_uploader("Public key (PEM, optional)", type=["pem"], key="ver_pem")
        up_json = st.file_uploader("Bundle JSON (optional)", type=["json"], key="ver_json")
        run = st.button("Run verification", type="primary", disabled=(up_pdf is None))
    with c2:
        st.write("**API_BASE**:", f"`{API_BASE}`")

    if run and up_pdf is not None:
        try:
            pub = up_pem.getvalue() if up_pem else None
            bjs = up_json.getvalue() if up_json else None
            res = _post_verify(up_pdf.getvalue(), pub, bjs)
            ok = res.get("ok")
            st.success("Verification PASSED") if ok else st.error("Verification FAILED")
            st.json(res)
            # Show the two hashes for quick human inspection
            st.code(f"printed report_sha256: {res.get('printed_sha256')}\ncomputed sha256:     {res.get('computed_sha256')}", language="text")
        except requests.HTTPError as e:
            detail = getattr(e, "response", None).text if getattr(e, "response", None) else ""
            st.error(f"Verify failed ({getattr(e.response, 'status_code', 'HTTP')}): {detail or ''}")
