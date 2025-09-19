# app/services/verify.py
from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # we’ll guard at runtime

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend
except Exception:
    hashes = None  # guard at runtime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
REPORTS_DIR  = DATA_DIR / "reports"


class VerifyInput(BaseModel):
    # Only when verifying against a known bundle on disk (optional)
    bundle_json_path: Optional[str] = None


@dataclass
class VerifyResult:
    ok: bool
    reasons: list[str]
    computed_sha256: Optional[str]
    printed_sha256: Optional[str]
    signature_ok: Optional[bool]
    bundle_match_ok: Optional[bool]
    parsed_fields: Dict[str, Any]


# ---- helpers ----
_SIG_RE = re.compile(
    r"report_sha256:\s*([a-fA-F0-9]{64}).*?signature\s*\(base64\):\s*([A-Za-z0-9+/=\s]+?)\s+signing_cert_subject:\s*(.+?)\s+signing_cert_pubkey_fingerprint:\s*([a-fA-F0-9]{64})",
    re.DOTALL
)

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _load_pdf_text(pdf_bytes: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. pip install pypdf")
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        try:
            text += page.extract_text() or ""
            text += "\n"
        except Exception:
            continue
    return text

def _parse_signature_block(pdf_text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Returns (printed_sha256, signature_b64, cert_subject, pubkey_fp) if found.
    """
    m = _SIG_RE.search(pdf_text)
    if not m:
        return None, None, None, None
    printed_sha = m.group(1).strip()
    sig_b64 = re.sub(r"\s+", "", m.group(2))
    subj = (m.group(3) or "").strip()
    fp = (m.group(4) or "").strip()
    return printed_sha, sig_b64, subj, fp

def _verify_signature(sig_b64: str, message_hex_sha256: str, public_pem: Optional[bytes]) -> Optional[bool]:
    """
    Verifies a base64 signature over the 32-byte digest.
    - Returns True/False if verification could be attempted, or None if crypto not available / no key.
    """
    if (hashes is None) or (public_pem is None):
        return None
    try:
        public_key = serialization.load_pem_public_key(public_pem, backend=default_backend())
        sig = base64.b64decode(sig_b64)
        digest = bytes.fromhex(message_hex_sha256)
        public_key.verify(
            sig,
            digest,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False

def _load_bundle(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _cross_check_bundle(pdf_text: str, bundle: Dict[str, Any]) -> Tuple[bool, list[str]]:
    """
    Lightweight cross-check:
      - report_id present in JSON equals the one seen on Page 1
      - Case ID, Investigator, and at least 1 evidence sha256 appear in PDF text
    """
    reasons: list[str] = []
    ok = True

    report_id = (bundle.get("report_id") or "").strip()
    case_id = (bundle.get("header", {}).get("case_id") or "").strip()
    investigator = (bundle.get("header", {}).get("investigator") or "").strip()
    evs = bundle.get("evidence", []) or []

    if report_id and (report_id not in pdf_text):
        ok = False
        reasons.append(f"report_id {report_id} not found in PDF text")

    if case_id and (str(case_id) not in pdf_text):
        ok = False
        reasons.append("Case ID not present in PDF text")

    if investigator and (investigator not in pdf_text):
        ok = False
        reasons.append("Investigator not present in PDF text")

    # Evidence: require at least one matching SHA-256
    any_hash = False
    for e in evs:
        h = (e.get("sha256") or "").lower()
        if h and (h in pdf_text.lower()):
            any_hash = True
            break
    if not any_hash and evs:
        ok = False
        reasons.append("No evidence sha256 from bundle found in PDF text")

    return ok, reasons


# ---- main entry ----
def verify_pdf(
    pdf_bytes: bytes,
    public_key_pem: Optional[bytes] = None,
    bundle_json: Optional[bytes] = None,
    bundle_json_path: Optional[str] = None,
) -> VerifyResult:
    reasons: list[str] = []

    # 1) Compute SHA-256 of the PDF
    computed_sha = _sha256_bytes(pdf_bytes)

    # 2) Parse PDF text for printed fields (report_sha256, signature, subject, fingerprint)
    pdf_text = _load_pdf_text(pdf_bytes)
    printed_sha, sig_b64, subject, pub_fp = _parse_signature_block(pdf_text)

    # 3) Compare hashes
    if printed_sha:
        if printed_sha != computed_sha:
            reasons.append("SHA-256 mismatch: printed report_sha256 != computed")
    else:
        reasons.append("Could not find report_sha256 in PDF")

    # 4) Verify signature if possible
    sig_ok: Optional[bool] = None
    if sig_b64:
        sig_ok = _verify_signature(sig_b64, computed_sha, public_key_pem)
        if sig_ok is False:
            reasons.append("Signature verification FAILED")
    else:
        reasons.append("Signature not found in PDF")

    # 5) Bundle cross-check (optional)
    bundle_ok: Optional[bool] = None
    bundle_dict: Optional[Dict[str, Any]] = None
    try:
        if bundle_json is not None:
            bundle_dict = json.loads(bundle_json.decode("utf-8"))
        elif bundle_json_path:
            bundle_dict = _load_bundle(bundle_json_path)
    except Exception:
        reasons.append("Could not read JSON bundle for cross-check")

    if bundle_dict is not None:
        bok, breasons = _cross_check_bundle(pdf_text, bundle_dict)
        bundle_ok = bok
        reasons.extend(breasons)

    # Final OK?
    overall_ok = True
    # We consider hash match mandatory when printed_sha exists
    if printed_sha and (printed_sha != computed_sha):
        overall_ok = False
    # If signature is present but verification failed, mark overall false
    if sig_b64 and (sig_ok is False):
        overall_ok = False
    # Bundle mismatch does not automatically fail unless you prefer strict mode

    return VerifyResult(
        ok=overall_ok,
        reasons=reasons,
        computed_sha256=computed_sha,
        printed_sha256=printed_sha,
        signature_ok=sig_ok,
        bundle_match_ok=bundle_ok,
        parsed_fields={
            "signing_cert_subject": subject,
            "signing_cert_pubkey_fingerprint": pub_fp,
        },
    )
