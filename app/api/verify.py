# app/api/verify.py
from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.services.verify import verify_pdf

router = APIRouter(prefix="/api/verify", tags=["verify"])


@router.post("/pdf")
async def verify_pdf_endpoint(
    pdf: UploadFile = File(..., description="Generated report PDF"),
    public_key_pem: UploadFile | None = File(None, description="PEM public key for signature verification"),
    bundle_json: UploadFile | None = File(None, description="JSON bundle for cross-check (optional)"),
):
    try:
        pdf_bytes = await pdf.read()
        pub_pem = await public_key_pem.read() if public_key_pem is not None else None
        bjson = await bundle_json.read() if bundle_json is not None else None

        result = verify_pdf(pdf_bytes, public_key_pem=pub_pem, bundle_json=bjson)
        return JSONResponse({
            "ok": result.ok,
            "reasons": result.reasons,
            "computed_sha256": result.computed_sha256,
            "printed_sha256": result.printed_sha256,
            "signature_ok": result.signature_ok,
            "bundle_match_ok": result.bundle_match_ok,
            "parsed": result.parsed_fields,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"verify_pdf failed: {e}")
