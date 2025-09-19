# app/services/report.py
from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fpdf import FPDF
from PIL import Image
import qrcode

# crypto (self-signed cert + signing)
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509.oid import NameOID

# --------------- constants / paths ---------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
ASSETS_DIR = REPORTS_DIR / "assets"
KEYS_DIR = DATA_DIR / "keys"
FONTS_DIR = DATA_DIR / "fonts"
DEJAVU_TTF = FONTS_DIR / "DejaVuSans.ttf"

GENERATING_SYSTEM_VERSION = "0.5.0"  # bump if needed

# --------------- dataclasses for structured inputs ---------------
@dataclass
class EvidenceItem:
    filename: str
    sha256: str
    ingest_time: str           # ISO8601
    camera_id: str             # user-supplied or "unknown"
    duration: Optional[float]  # seconds for videos; None for images

@dataclass
class FindingItem:
    time_window: str                   # "start_iso .. end_iso" or human string
    track_id: Optional[int]
    object_type: str                   # e.g., "person", "car", "motion"
    representative_frame_path: Optional[str]   # image file path (png/jpg)
    bbox: Optional[Tuple[int, int, int, int]]  # x,y,w,h
    matched_offender_id: Optional[str]         # from faces search
    matched_offender_name: Optional[str]       # if verified
    similarity_score: Optional[float]
    verification_status: str                   # "verified"/"unverified"/"unknown"

@dataclass
class ForensicBlock:
    metadata_summary: Dict[str, Any]   # EXIF/ffprobe highlights
    tamper_flags: List[str]            # short notes
    ela_thumbnails: List[str]          # paths to ELA images to embed
    deepfake_score: Optional[float]    # heuristic; disclaimer shown in PDF

@dataclass
class ReportHeader:
    case_id: str
    investigator: str
    station_unit: str
    contact: str
    case_notes: str

@dataclass
class ReportSpec:
    header: ReportHeader
    evidence: List[EvidenceItem]
    findings: List[FindingItem]
    forensics: ForensicBlock
    bundle_json: Dict[str, Any]  # extra machine-readable fields

# --------------- utils ---------------
def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    FONTS_DIR.mkdir(parents=True, exist_ok=True)

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _resize_fit(img_path: Path, max_w: int, max_h: int) -> Path:
    """Return a temp path to a resized copy (keeps aspect)."""
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((max_w, max_h))
    out = ASSETS_DIR / f"{img_path.stem}.fit_{max_w}x{max_h}.jpg"
    img.save(out, "JPEG", quality=90)
    return out

def _qr_png_from_text(text: str, name: str) -> Path:
    img = qrcode.make(text)
    out = ASSETS_DIR / f"{name}.qr.png"
    img.save(out)
    return out

# ---------- text safety for fonts ----------
def _sanitize_latin1(s: str) -> str:
    """
    Replace common Unicode punctuation with ASCII fallbacks, then
    map anything still outside Latin-1 to '?' (so core fonts won't crash).
    """
    if s is None:
        return ""
    replacements = {
        "—": "-", "–": "-", "―": "-",  # dashes
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "•": "*", "…": "...",
        "×": "x",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s.encode("latin-1", errors="replace").decode("latin-1")

def _safe_text(s: str, unicode_ok: bool) -> str:
    return s if unicode_ok else _sanitize_latin1(s or "")

def _fit_cell_text(pdf: FPDF, s: str, max_w_mm: float, unicode_ok: bool, ellipsis: str = "…") -> str:
    """
    Ensure the text fits into a single-cell width by truncating with ellipsis.
    Uses fpdf2's current font metrics via get_string_width().
    """
    s = _safe_text(s or "", unicode_ok)
    # small inner padding so we don't sit right on the border
    pad = 2.0
    if pdf.get_string_width(s) <= max(0.0, max_w_mm - pad):
        return s
    # iteratively trim
    base = s
    while base and pdf.get_string_width(base + ellipsis) > max(0.0, max_w_mm - pad):
        base = base[:-1]
    return (base + ellipsis) if base else ellipsis

# --------------- self-signed cert for demo signing ---------------
def _cert_files() -> Tuple[Path, Path]:
    key_pem = KEYS_DIR / "report_signing_key.pem"
    cert_pem = KEYS_DIR / "report_signing_cert.pem"
    return key_pem, cert_pem

def _ensure_self_signed_cert(subject_cn: str = "Demo Report Signer") -> Tuple[Path, Path]:
    key_pem, cert_pem = _cert_files()
    if key_pem.exists() and cert_pem.exists():
        return key_pem, cert_pem

    # generate RSA key
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, subject_cn),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AI CCTV & DMFT (Hackathon)"),
        x509.NameAttribute(NameOID.COUNTRY_NAME, "IN"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(datetime.utcnow().replace(year=datetime.utcnow().year + 2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(private_key=key, algorithm=hashes.SHA256())
    )

    _ensure_dirs()
    with open(key_pem, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    with open(cert_pem, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    return key_pem, cert_pem

def _sign_bytes(data: bytes) -> Dict[str, str]:
    key_pem, cert_pem = _ensure_self_signed_cert()
    with open(key_pem, "rb") as f:
        key = serialization.load_pem_private_key(f.read(), password=None)
    with open(cert_pem, "rb") as f:
        cert = x509.load_pem_x509_certificate(f.read())

    sig = key.sign(
        data,
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    sig_b64 = base64.b64encode(sig).decode("ascii")

    subj = cert.subject.rfc4514_string()

    spki = cert.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    fp = hashlib.sha256(spki).hexdigest()

    return {
        "signature_b64": sig_b64,
        "signing_cert_subject": subj,
        "signing_cert_pubkey_fingerprint": fp,
    }

# --------------- PDF builder ---------------
class _PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._unicode_font_loaded = False

    def load_unicode_font(self):
        """Try to register and select DejaVuSans for Unicode support."""
        if DEJAVU_TTF.exists():
            self.add_font("DejaVu", "", str(DEJAVU_TTF), uni=True)
            self.add_font("DejaVu", "B", str(DEJAVU_TTF), uni=True)  # bold maps to same file
            self.set_font("DejaVu", "", 10)
            self._unicode_font_loaded = True
        else:
            self.set_font("Helvetica", "", 10)
            self._unicode_font_loaded = False

    def header(self):
        # Keep ASCII here so it renders even without Unicode font
        title = "AI CCTV & Digital Media Forensic Report"
        subtitle = "For hackathon demonstration - not a legal forensic document"
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, title, ln=1, align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, subtitle, ln=1, align="C")
        self.ln(2)
        self.set_draw_color(180, 180, 180)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

def _kv(pdf: _PDF, k: str, v: str, unicode_ok: bool, k_w: int = 45):
    pdf.set_font(("DejaVu" if pdf._unicode_font_loaded else "Helvetica"), "B", 10)
    pdf.cell(k_w, 6, _safe_text(k, unicode_ok))
    pdf.set_font(("DejaVu" if pdf._unicode_font_loaded else "Helvetica"), "", 10)
    pdf.multi_cell(0, 6, _safe_text(v, unicode_ok))

def _add_table_header(pdf: _PDF, headers: List[str], widths: List[int], unicode_ok: bool):
    pdf.set_font(("DejaVu" if pdf._unicode_font_loaded else "Helvetica"), "B", 9)
    for h, w in zip(headers, widths):
        txt = _fit_cell_text(pdf, h, float(w), unicode_ok)
        pdf.cell(w, 7, txt, border=1, align="C")
    pdf.ln(7)
    pdf.set_font(("DejaVu" if pdf._unicode_font_loaded else "Helvetica"), "", 9)


def _add_table_row(pdf: _PDF, row: List[str], widths: List[int], unicode_ok: bool):
    for c, w in zip(row, widths):
        txt = _fit_cell_text(pdf, c, float(w), unicode_ok)
        pdf.cell(w, 6, txt, border=1)
    pdf.ln(6)


# --------------- public API ---------------
def generate_report(spec: ReportSpec) -> Dict[str, Any]:
    """
    Build a PDF + JSON sidecar according to organizer fields, plus extras.
    Returns dict with paths and signature info.
    """
    _ensure_dirs()

    # ---- IDs / timestamps ----
    report_id = hashlib.sha1(os.urandom(16)).hexdigest()  # okay for demo
    generation_time_utc = datetime.now(timezone.utc).isoformat()

    # ---- JSON bundle (QR points to this local path in demo) ----
    bundle = {
        "report_id": report_id,
        "generation_time_utc": generation_time_utc,
        "generating_system_version": GENERATING_SYSTEM_VERSION,
        "header": asdict(spec.header),
        "evidence": [asdict(e) for e in spec.evidence],
        "findings": [asdict(f) for f in spec.findings],
        "forensics": asdict(spec.forensics),
    }
    try:
        bundle.update(spec.bundle_json or {})
    except Exception:
        pass

    json_path = REPORTS_DIR / f"{report_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    # ---- QR (points to the JSON bundle path for demo) ----
    qr_path = _qr_png_from_text(text=str(json_path), name=f"qr_{report_id}")

    # ---- PDF init ----
    pdf = _PDF(orientation="P", unit="mm", format="A4")
    pdf.set_creator("AI CCTV & DMFT — Report Generator")
    pdf.set_title(f"Report {report_id}")
    pdf.set_author(spec.header.investigator or "unknown")
    pdf.add_page()
    pdf.load_unicode_font()
    unicode_ok = pdf._unicode_font_loaded

    # ---- Header (Case / Investigator) ----
    pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "B", 11)
    pdf.cell(0, 7, _safe_text("Case Header", unicode_ok), ln=1)
    _kv(pdf, "Report ID", report_id, unicode_ok)
    _kv(pdf, "Generated (UTC)", generation_time_utc, unicode_ok)
    _kv(pdf, "System Version", GENERATING_SYSTEM_VERSION, unicode_ok)

    pdf.ln(2)
    _kv(pdf, "Case ID", spec.header.case_id, unicode_ok)
    _kv(pdf, "Investigator", spec.header.investigator, unicode_ok)
    _kv(pdf, "Station/Unit", spec.header.station_unit, unicode_ok)
    _kv(pdf, "Contact", spec.header.contact, unicode_ok)
    if (spec.header.case_notes or "").strip():
        _kv(pdf, "Case Notes", spec.header.case_notes, unicode_ok)

    # QR on the right
    y_now = pdf.get_y()
    x_qr = 155
    try:
        qr_fit = _resize_fit(Path(qr_path), 45, 45)
        pdf.image(str(qr_fit), x=x_qr, y=y_now-18 if y_now > 18 else 18, w=40)
    except Exception:
        pass

    pdf.ln(4)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # ---- Evidence table ----
    pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "B", 11)
    pdf.cell(0, 7, _safe_text("Evidence List", unicode_ok), ln=1)
    headers = ["Filename", "SHA-256", "Ingest time", "Camera ID", "Duration (s)"]
    widths = [45, 65, 35, 25, 20]
    _add_table_header(pdf, headers, widths, unicode_ok)
    for e in spec.evidence:
        _add_table_row(pdf, [
            e.filename,
            (e.sha256[:10] + "…"),
            e.ingest_time,
            (e.camera_id or "unknown"),
            (f"{e.duration:.1f}" if e.duration else "-"),
        ], widths, unicode_ok)

    pdf.ln(3)

    # ---- Findings ----
    pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "B", 11)
    pdf.cell(0, 7, _safe_text("Findings", unicode_ok), ln=1)
    if not spec.findings:
        pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "", 10)
        pdf.multi_cell(0, 6, _safe_text("No detection/face findings selected.", unicode_ok))
    else:
        for idx, f in enumerate(spec.findings, 1):
            pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "B", 10)
            pdf.cell(0, 6, _safe_text(f"Event #{idx}", unicode_ok), ln=1)
            pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "", 10)
            _kv(pdf, "Time window", f.time_window, unicode_ok)
            _kv(pdf, "Track ID", (str(f.track_id) if f.track_id is not None else "-"), unicode_ok)
            _kv(pdf, "Object type", f.object_type, unicode_ok)
            _kv(pdf, "BBox (x,y,w,h)", (str(f.bbox) if f.bbox else "-"), unicode_ok)
            _kv(pdf, "Matched offender (ID)", (f.matched_offender_id or "-"), unicode_ok)
            _kv(pdf, "Matched offender (Name)", (f.matched_offender_name or "-"), unicode_ok)
            _kv(pdf, "Similarity score", (f"{f.similarity_score:.3f}" if f.similarity_score is not None else "-"), unicode_ok)
            _kv(pdf, "Verification status", (f.verification_status or "unknown"), unicode_ok)
            # representative image (fit)
            if f.representative_frame_path:
                try:
                    fit = _resize_fit(Path(f.representative_frame_path), 170, 80)
                    pdf.image(str(fit), w=170)
                except Exception:
                    pdf.multi_cell(0, 6, _safe_text(f"(Could not render image: {f.representative_frame_path})", unicode_ok))
            pdf.ln(2)

    pdf.ln(2)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # ---- Forensics (metadata + ELA + deepfake heuristic) ----
    pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "B", 11)
    pdf.cell(0, 7, _safe_text("Forensics Summary", unicode_ok), ln=1)

    # metadata highlights
    pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "", 10)
    meta_txt = json.dumps(spec.forensics.metadata_summary, ensure_ascii=False, indent=2)
    meta_body = meta_txt[:1800] + (" …(truncated)" if len(meta_txt) > 1800 else "")
    pdf.multi_cell(0, 5, _safe_text(meta_body, unicode_ok))

    # tamper flags
    if spec.forensics.tamper_flags:
        pdf.ln(2)
        pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "B", 10)
        pdf.cell(0, 6, _safe_text("Tamper checks:", unicode_ok), ln=1)
        pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "", 10)
        for flag in spec.forensics.tamper_flags:
            pdf.multi_cell(0, 5, _safe_text(f"• {flag}", unicode_ok))

    # ELA thumbnails grid
    if spec.forensics.ela_thumbnails:
        pdf.ln(2)
        pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "B", 10)
        pdf.cell(0, 6, _safe_text("ELA thumbnails:", unicode_ok), ln=1)
        pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "", 10)
        max_w = 60
        x0, y0 = pdf.get_x(), pdf.get_y()
        x, y = x0, y0
        per_row = 3
        i = 0
        for p in spec.forensics.ela_thumbnails[:9]:
            try:
                fit = _resize_fit(Path(p), max_w, 60)
                pdf.image(str(fit), x=x, y=y, w=max_w)
            except Exception:
                pdf.text(x, y + 5, _safe_text(f"(img fail: {p})", unicode_ok))
            i += 1
            if i % per_row == 0:
                y += 62
                x = x0
            else:
                x += max_w + 4
        pdf.set_y(y + 65)

    # deepfake heuristic (with clear disclaimer)
    pdf.ln(2)
    pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "B", 10)
    pdf.cell(0, 6, _safe_text("Deepfake heuristic score (experimental):", unicode_ok), ln=1)
    pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "", 10)
    if spec.forensics.deepfake_score is None:
        pdf.multi_cell(0, 5, _safe_text("N/A", unicode_ok))
    else:
        pdf.multi_cell(
            0, 5,
            _safe_text(
                f"{spec.forensics.deepfake_score:.2f} (0=low suspicion .. 1=high suspicion) - "
                "heuristic only; NOT a definitive detector.",
                unicode_ok
            )
        )

    pdf.ln(3)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    # ---- Signatures (hash + signature + cert info) ----
    pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "B", 11)
    pdf.cell(0, 7, _safe_text("Signatures", unicode_ok), ln=1)
    pdf.set_font(("DejaVu" if unicode_ok else "Helvetica"), "", 10)
    pdf.multi_cell(0, 5, _safe_text("The following fields allow integrity verification of this PDF in demo mode.", unicode_ok))

    # Temporarily output PDF to bytes to hash and sign
    tmp_pdf = REPORTS_DIR / f"{report_id}.tmp.pdf"
    pdf.output(str(tmp_pdf))
    pdf_bytes = tmp_pdf.read_bytes()
    report_sha256 = _sha256_bytes(pdf_bytes)

    sig_info = _sign_bytes(bytes.fromhex(report_sha256))
    signature_b64 = sig_info["signature_b64"]
    signing_cert_subject = sig_info["signing_cert_subject"]
    signing_cert_pubkey_fingerprint = sig_info["signing_cert_pubkey_fingerprint"]

    _kv(pdf, "report_sha256", report_sha256, unicode_ok)
    _kv(pdf, "signature (base64)", signature_b64[:96] + "…", unicode_ok)
    _kv(pdf, "signing_cert_subject", signing_cert_subject, unicode_ok)
    _kv(pdf, "signing_cert_pubkey_fingerprint", signing_cert_pubkey_fingerprint, unicode_ok)

    # Final write
    out_pdf = REPORTS_DIR / f"{report_id}.pdf"
    pdf.output(str(out_pdf))
    tmp_pdf.unlink(missing_ok=True)

    # return summary
    return {
        "report_id": report_id,
        "pdf_path": str(out_pdf),
        "json_path": str(json_path),
        "qr_path": str(qr_path),
        "report_sha256": report_sha256,
        "signature_b64": signature_b64,
        "signing_cert_subject": signing_cert_subject,
        "signing_cert_pubkey_fingerprint": signing_cert_pubkey_fingerprint,
        "generation_time_utc": generation_time_utc,
        "generating_system_version": GENERATING_SYSTEM_VERSION,
    }
