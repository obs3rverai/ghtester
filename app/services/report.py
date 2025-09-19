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

from PIL import Image
import qrcode

# ReportLab
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# crypto (self-signed cert + signing)
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.x509.oid import NameOID

# ---------------- paths / constants ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = DATA_DIR / "reports"
ASSETS_DIR = REPORTS_DIR / "assets"
KEYS_DIR = DATA_DIR / "keys"
FONTS_DIR = DATA_DIR / "fonts"
DEJAVU_TTF = FONTS_DIR / "DejaVuSans.ttf"

GENERATING_SYSTEM_VERSION = "0.5.0"

# Evidence table widths (in points; 1 mm ≈ 2.835 pts). Keep sum < page width minus margins.
EVIDENCE_COL_WIDTHS = [
    60*mm,  # Filename
    65*mm,  # SHA-256
    40*mm,  # Ingest time
    30*mm,  # Camera ID
    25*mm,  # Duration
]

# ---------------- dataclasses ----------------
@dataclass
class EvidenceItem:
    filename: str
    sha256: str
    ingest_time: str           # ISO8601
    camera_id: str
    duration: Optional[float]

@dataclass
class FindingItem:
    time_window: str
    track_id: Optional[int]
    object_type: str
    representative_frame_path: Optional[str]
    bbox: Optional[Tuple[int, int, int, int]]
    matched_offender_id: Optional[str]
    matched_offender_name: Optional[str]
    similarity_score: Optional[float]
    verification_status: str

@dataclass
class ForensicBlock:
    metadata_summary: Dict[str, Any]
    tamper_flags: List[str]
    ela_thumbnails: List[str]
    deepfake_score: Optional[float]

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
    bundle_json: Dict[str, Any]

# ---------------- utils ----------------
def _ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    FONTS_DIR.mkdir(parents=True, exist_ok=True)

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _resize_fit(img_path: Path, max_w_px: int, max_h_px: int) -> Path:
    """Resize to fit within WxH (pixels), save to ASSETS, return path."""
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((max_w_px, max_h_px))
    out = ASSETS_DIR / f"{img_path.stem}.fit_{max_w_px}x{max_h_px}.jpg"
    img.save(out, "JPEG", quality=90)
    return out

def _qr_png_from_text(text: str, name: str) -> Path:
    img = qrcode.make(text)
    out = ASSETS_DIR / f"{name}.qr.png"
    img.save(out)
    return out

def _sanitize_ascii(s: str) -> str:
    """ASCII-safe fallback when Unicode font missing (replace smart quotes, dash, bullets, etc.)."""
    if s is None:
        return ""
    repl = {
        "—": "-", "–": "-", "―": "-", "…": "...", "•": "*",
        "“": '"', "”": '"', "„": '"', "‟": '"',
        "‘": "'", "’": "'", "‚": "'", "‛": "'",
        "×": "x",
    }
    for k, v in repl.items(): s = s.replace(k, v)
    return s.encode("ascii", errors="replace").decode("ascii")

def _safe(s: Any, unicode_ok: bool) -> str:
    s = "" if s is None else str(s)
    return s if unicode_ok else _sanitize_ascii(s)

# ---------------- font setup ----------------
def _setup_fonts() -> Tuple[str, bool]:
    """
    Register DejaVuSans if present; return primary font name and a unicode_ok flag.
    """
    try:
        if DEJAVU_TTF.exists():
            pdfmetrics.registerFont(TTFont("DejaVu", str(DEJAVU_TTF)))
            return "DejaVu", True
    except Exception:
        pass
    # Fallback to Helvetica (ASCII-safe text only)
    return "Helvetica", False

# ---------------- signing (demo) ----------------
def _cert_files() -> Tuple[Path, Path]:
    return (KEYS_DIR / "report_signing_key.pem", KEYS_DIR / "report_signing_cert.pem")

def _ensure_self_signed_cert(subject_cn: str = "Demo Report Signer") -> Tuple[Path, Path]:
    key_pem, cert_pem = _cert_files()
    if key_pem.exists() and cert_pem.exists():
        return key_pem, cert_pem

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
    sig = key.sign(data, padding.PKCS1v15(), hashes.SHA256())
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

# ---------------- PDF builder ----------------
def _styles(primary_font: str) -> Dict[str, ParagraphStyle]:
    ss = getSampleStyleSheet()
    base = ParagraphStyle(
        "Base",
        parent=ss["Normal"],
        fontName=primary_font,
        fontSize=9,
        leading=11,
        spaceAfter=2,
    )
    header = ParagraphStyle("Header", parent=base, fontSize=12, leading=14, spaceAfter=6)
    h2 = ParagraphStyle("H2", parent=base, fontSize=11, leading=13, spaceBefore=6, spaceAfter=4, )
    key = ParagraphStyle("Key", parent=base, fontName=primary_font, fontSize=9, leading=11, )
    val = ParagraphStyle("Val", parent=base, fontName=primary_font, fontSize=9, leading=11, )
    mono = ParagraphStyle("Mono", parent=base, fontName=primary_font, fontSize=8, leading=10)
    small = ParagraphStyle("Small", parent=base, fontSize=8, leading=9)
    return {"base": base, "header": header, "h2": h2, "key": key, "val": val, "mono": mono, "small": small}

def _kv_paragraph(k: str, v: str, st: Dict[str, ParagraphStyle], unicode_ok: bool) -> Paragraph:
    return Paragraph(f"<b>{_safe(k, unicode_ok)}:</b> {_safe(v, unicode_ok)}", st["base"])

def _make_table(data: List[List[str | Paragraph]], col_widths: List[float], primary_font: str) -> Table:
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), primary_font),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("LEADING", (0,0), (-1,-1), 11),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F0F0F6")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ALIGN", (-1,1), (-1,-1), "RIGHT"),  # duration column right-aligned
    ]))
    return tbl

def _header_footer(canvas, doc, title_text="AI CCTV & Digital Media Forensic Report"):
    canvas.saveState()
    # header
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawCentredString(A4[0]/2, A4[1]-12*mm, title_text)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(A4[0]/2, A4[1]-16*mm, "For hackathon demonstration - not a legal forensic document")
    # footer
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.drawCentredString(A4[0]/2, 10*mm, f"Page {doc.page}")
    canvas.restoreState()

# ---------------- public API ----------------
def generate_report(spec: ReportSpec) -> Dict[str, Any]:
    """
    Build a PDF + JSON sidecar according to organizer fields, plus extras:
    - Header (Case/Investigator/Station/Contact/Notes)
    - Evidence table (wrapped cells)
    - Findings (with thumbnails)
    - Forensics (metadata, ELA thumbs, deepfake heuristic with disclaimer)
    - Signatures (hash + demo signature)
    """
    _ensure_dirs()

    # IDs / timestamps
    report_id = hashlib.sha1(os.urandom(16)).hexdigest()
    generation_time_utc = datetime.now(timezone.utc).isoformat()

    # JSON bundle (what QR points to for demo)
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

    # QR to the JSON path (demo)
    qr_path = _qr_png_from_text(str(json_path), f"qr_{report_id}")

    # Fonts / styles
    primary_font, unicode_ok = _setup_fonts()
    st = _styles(primary_font)

    # Doc
    out_pdf = REPORTS_DIR / f"{report_id}.pdf"
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm, topMargin=25*mm, bottomMargin=15*mm,
        title=f"Report {report_id}",
        author=spec.header.investigator or "unknown",
    )
    story: List[Any] = []

    # ----- Header block
    story.append(Paragraph("Case Header", st["h2"]))
    story.append(_kv_paragraph("Report ID", report_id, st, unicode_ok))
    story.append(_kv_paragraph("Generated (UTC)", generation_time_utc, st, unicode_ok))
    story.append(_kv_paragraph("System Version", GENERATING_SYSTEM_VERSION, st, unicode_ok))
    story.append(Spacer(1, 3*mm))
    story.append(_kv_paragraph("Case ID", spec.header.case_id, st, unicode_ok))
    story.append(_kv_paragraph("Investigator", spec.header.investigator, st, unicode_ok))
    story.append(_kv_paragraph("Station/Unit", spec.header.station_unit, st, unicode_ok))
    story.append(_kv_paragraph("Contact", spec.header.contact, st, unicode_ok))
    if (spec.header.case_notes or "").strip():
        story.append(_kv_paragraph("Case Notes", spec.header.case_notes, st, unicode_ok))

    # QR on the right (small trick: put it as an image then let text wrap above it)
    try:
        qr_fit = _resize_fit(Path(qr_path), 400, 400)  # ~400 px square
        story.append(Spacer(1, 2*mm))
        story.append(RLImage(str(qr_fit), width=35*mm, height=35*mm))
    except Exception:
        pass

    story.append(Spacer(1, 4*mm))

    # ----- Evidence table (wrapped, never overflows)
    story.append(Paragraph("Evidence List", st["h2"]))
    headers = ["Filename", "SHA-256", "Ingest time", "Camera ID", "Duration (s)"]
    header_row = [Paragraph(h, st["base"]) for h in headers]
    rows: List[List[Any]] = [header_row]
    for e in spec.evidence:
        rows.append([
            Paragraph(_safe(e.filename, unicode_ok), st["base"]),
            Paragraph(_safe(e.sha256, unicode_ok), st["mono"]),
            Paragraph(_safe(e.ingest_time, unicode_ok), st["base"]),
            Paragraph(_safe(e.camera_id or "unknown", unicode_ok), st["base"]),
            Paragraph(_safe(f"{e.duration:.1f}" if e.duration else "-", unicode_ok), st["base"]),
        ])
    story.append(_make_table(rows, EVIDENCE_COL_WIDTHS, primary_font))
    story.append(Spacer(1, 4*mm))

    # ----- Findings
    story.append(Paragraph("Findings", st["h2"]))
    if not spec.findings:
        story.append(Paragraph("No detection/face findings selected.", st["base"]))
    else:
        for i, f in enumerate(spec.findings, 1):
            story.append(Paragraph(f"Event #{i}", st["base"]))
            story.append(_kv_paragraph("Time window", f.time_window, st, unicode_ok))
            story.append(_kv_paragraph("Track ID", "-" if f.track_id is None else str(f.track_id), st, unicode_ok))
            story.append(_kv_paragraph("Object type", f.object_type, st, unicode_ok))
            story.append(_kv_paragraph("BBox (x,y,w,h)", "-" if not f.bbox else str(f.bbox), st, unicode_ok))
            story.append(_kv_paragraph("Matched offender (ID)", f.matched_offender_id or "-", st, unicode_ok))
            story.append(_kv_paragraph("Matched offender (Name)", f.matched_offender_name or "-", st, unicode_ok))
            story.append(_kv_paragraph("Similarity score", "-" if f.similarity_score is None else f"{f.similarity_score:.3f}", st, unicode_ok))
            story.append(_kv_paragraph("Verification status", f.verification_status or "unknown", st, unicode_ok))
            if f.representative_frame_path:
                try:
                    fit = _resize_fit(Path(f.representative_frame_path), 1400, 700)  # wide preview
                    story.append(Spacer(1, 1*mm))
                    # width ~170mm -> keep aspect
                    story.append(RLImage(str(fit), width=170*mm, height=None))
                except Exception:
                    story.append(Paragraph(f"(Could not render image: {f.representative_frame_path})", st["small"]))
            story.append(Spacer(1, 3*mm))
    story.append(Spacer(1, 3*mm))

    # ----- Forensics
    story.append(Paragraph("Forensics Summary", st["h2"]))
    meta_txt = json.dumps(spec.forensics.metadata_summary, ensure_ascii=False, indent=2)
    # Limit huge dumps (platypus can still handle long paragraphs, but keep readable)
    if len(meta_txt) > 4000:
        meta_txt = meta_txt[:4000] + " …(truncated)"
    story.append(Paragraph(_safe(meta_txt, unicode_ok).replace("\n", "<br/>"), st["mono"]))
    if spec.forensics.tamper_flags:
        story.append(Spacer(1, 1*mm))
        story.append(Paragraph("<b>Tamper checks:</b>", st["base"]))
        for fl in spec.forensics.tamper_flags:
            story.append(Paragraph("• " + _safe(fl, unicode_ok), st["base"]))
    if spec.forensics.ela_thumbnails:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("ELA thumbnails:", st["base"]))
        # place up to 3 per row
        thumbs: List[RLImage] = []
        for p in spec.forensics.ela_thumbnails[:9]:
            try:
                fit = _resize_fit(Path(p), 600, 300)
                thumbs.append(RLImage(str(fit), width=60*mm, height=None))
            except Exception:
                pass
        # arrange in rows of 3 by simple flow
        for i in range(0, len(thumbs), 3):
            row = thumbs[i:i+3]
            # Use a table for neat grids
            t = Table([[*row]], colWidths=[60*mm]*len(row))
            t.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "LEFT"), ("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
            story.append(t)
            story.append(Spacer(1, 2*mm))

    story.append(Spacer(1, 2*mm))
    story.append(Paragraph("<b>Deepfake heuristic score (experimental):</b>", st["base"]))
    if spec.forensics.deepfake_score is None:
        story.append(Paragraph("N/A", st["base"]))
    else:
        story.append(Paragraph(
            _safe(f"{spec.forensics.deepfake_score:.2f} (0=low suspicion .. 1=high suspicion) - heuristic only; NOT a definitive detector.", unicode_ok),
            st["base"]
        ))

    story.append(Spacer(1, 4*mm))

    # ----- Signatures
    story.append(Paragraph("Signatures", st["h2"]))
    story.append(Paragraph("The following fields allow integrity verification of this PDF in demo mode.", st["base"]))

    # Build once to file to compute hash
    doc.build(story, onFirstPage=lambda c, d: _header_footer(c, d),
              onLaterPages=lambda c, d: _header_footer(c, d))

    pdf_bytes = out_pdf.read_bytes()
    report_sha256 = _sha256_bytes(pdf_bytes)
    sig_info = _sign_bytes(bytes.fromhex(report_sha256))
    signature_b64 = sig_info["signature_b64"]
    signing_cert_subject = sig_info["signing_cert_subject"]
    signing_cert_pubkey_fingerprint = sig_info["signing_cert_pubkey_fingerprint"]

    # Append signature page
    story2: List[Any] = []
    story2.append(Paragraph("Signatures (continued)", st["h2"]))
    story2.append(Paragraph(f"<b>report_sha256:</b> {_safe(report_sha256, unicode_ok)}", st["mono"]))
    story2.append(Paragraph(f"<b>signature (base64):</b> {_safe(signature_b64[:96] + '…', unicode_ok)}", st["mono"]))
    story2.append(Paragraph(f"<b>signing_cert_subject:</b> {_safe(signing_cert_subject, unicode_ok)}", st["mono"]))
    story2.append(Paragraph(f"<b>signing_cert_pubkey_fingerprint:</b> {_safe(signing_cert_pubkey_fingerprint, unicode_ok)}", st["mono"]))

    # Rebuild with signature page appended
    story.append(PageBreak())
    story.extend(story2)
    doc.build(story, onFirstPage=lambda c, d: _header_footer(c, d),
              onLaterPages=lambda c, d: _header_footer(c, d))

    # Return summary
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
