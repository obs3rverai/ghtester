# app/services/report.py
from __future__ import annotations

import base64
import hashlib
import io
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

GENERATING_SYSTEM_VERSION = "0.5.1"

# Evidence table widths (sum must be 180 mm for A4 with 15 mm margins)
# Filename, SHA-256, Ingest time, Camera ID, Duration
EVIDENCE_COL_WIDTHS = [58*mm, 52*mm, 30*mm, 25*mm, 15*mm]  # 58+52+30+25+15 = 180

# ---------------- dataclasses ----------------
@dataclass
class EvidenceItem:
    filename: str
    sha256: str
    ingest_time: str
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
    if s is None: return ""
    repl = {
        "—":"-","–":"-","―":"-","…":"...","•":"*",
        "“":'"',"”":'"',"„":'"',"‟":'"',
        "‘":"'","’":"'","‚":"'","‛":"'", "×":"x",
    }
    for k,v in repl.items(): s = s.replace(k,v)
    return s.encode("ascii", errors="replace").decode("ascii")

def _safe(s: Any, unicode_ok: bool) -> str:
    s = "" if s is None else str(s)
    return s if unicode_ok else _sanitize_ascii(s)

# ---------------- font setup ----------------
def _setup_fonts() -> Tuple[str, bool]:
    try:
        if DEJAVU_TTF.exists():
            pdfmetrics.registerFont(TTFont("DejaVu", str(DEJAVU_TTF)))
            return "DejaVu", True
    except Exception:
        pass
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
    cert = (x509.CertificateBuilder()
            .subject_name(subject).issuer_name(issuer).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow().replace(year=datetime.utcnow().year + 2))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(private_key=key, algorithm=hashes.SHA256()))
    with open(key_pem, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()))
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

# ---------------- styles & header/footer ----------------
def _styles(primary_font: str) -> Dict[str, ParagraphStyle]:
    ss = getSampleStyleSheet()
    base = ParagraphStyle(
        "Base", parent=ss["Normal"], fontName=primary_font,
        fontSize=9, leading=11, spaceAfter=2, wordWrap="CJK"  # allow wrap inside long tokens
    )
    header = ParagraphStyle("Header", parent=base, fontSize=12, leading=14, spaceAfter=6)
    h2 = ParagraphStyle("H2", parent=base, fontSize=11, leading=13, spaceBefore=6, spaceAfter=4)
    mono = ParagraphStyle("Mono", parent=base, fontName=primary_font, fontSize=8, leading=10, wordWrap="CJK")
    small = ParagraphStyle("Small", parent=base, fontSize=8, leading=9, wordWrap="CJK")
    return {"base": base, "header": header, "h2": h2, "mono": mono, "small": small}

def _kv_p(k: str, v: str, st: Dict[str, ParagraphStyle], unicode_ok: bool) -> Paragraph:
    return Paragraph(f"<b>{_safe(k, unicode_ok)}:</b> {_safe(v, unicode_ok)}", st["base"])

def _make_table(data: List[List[str|Paragraph]], col_widths: List[float], primary_font: str) -> Table:
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), primary_font),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("LEADING", (0,0), (-1,-1), 11),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BOX", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F0F0F6")),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("ALIGN", (-1,1), (-1,-1), "RIGHT"),  # duration column right aligned
    ]))
    return tbl

def _header_footer(canvas, doc, title_text="AI CCTV & Digital Media Forensic Report"):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawCentredString(A4[0]/2, A4[1]-12*mm, title_text)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(A4[0]/2, A4[1]-16*mm, "For hackathon demonstration - not a legal forensic document")
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.drawCentredString(A4[0]/2, 10*mm, f"Page {doc.page}")
    canvas.restoreState()

# ---------------- concise metadata summary ----------------
def _human_size(n: Optional[int]) -> str:
    try:
        n = int(n or 0)
    except Exception:
        return "-"
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}"
        n /= 1024.0
    return "-"

def _summarize_metadata(md: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    out["path"]  = Path(str(md.get("path",""))).name
    out["kind"]  = str(md.get("kind","-"))
    out["mime"]  = str(md.get("mime","-"))
    out["sha256"] = str(md.get("sha256","-"))[:16] + "…"
    size = md.get("size_bytes") or md.get("details",{}).get("format",{}).get("size")
    out["size"]  = _human_size(size)
    det = md.get("details",{})
    out["duration"] = "-"
    out["fps"]      = "-"
    out["codec"]    = "-"
    out["res"]      = "-"
    out["bitrate"]  = "-"
    # video hints
    try:
        if "duration_sec" in det:
            out["duration"] = f"{float(det['duration_sec']):.2f}s"
        elif "ffprobe" in det and "format" in det["ffprobe"]:
            dur = det["ffprobe"]["format"].get("duration")
            if dur: out["duration"] = f"{float(dur):.2f}s"
        if "fps" in det:
            out["fps"] = f"{float(det['fps']):.2f}"
        streams = det.get("ffprobe",{}).get("streams",[])
        vstr = next((s for s in streams if s.get("codec_type")=="video"), {})
        if vstr:
            w, h = vstr.get("width"), vstr.get("height")
            if w and h: out["res"] = f"{w}x{h}"
            if vstr.get("codec_name"): out["codec"] = vstr["codec_name"]
        br = det.get("bit_rate") or det.get("ffprobe",{}).get("format",{}).get("bit_rate")
        if br: out["bitrate"] = f"{int(br)//1000} kbps"
    except Exception:
        pass
    return out

# ---------------- build story ----------------
def _styleset(primary_font: str) -> Dict[str, ParagraphStyle]:
    return _styles(primary_font)

def _build_story(spec: ReportSpec, st: Dict[str, ParagraphStyle], primary_font: str, unicode_ok: bool, qr_path: Path) -> List[Any]:
    story: List[Any] = []
    # Case header
    story.append(Paragraph("Case Header", st["h2"]))
    story.append(_kv_p("Report ID", spec.bundle_json.get("report_id",""), st, unicode_ok))
    story.append(_kv_p("Generated (UTC)", spec.bundle_json.get("generation_time_utc",""), st, unicode_ok))
    story.append(_kv_p("System Version", GENERATING_SYSTEM_VERSION, st, unicode_ok))
    story.append(Spacer(1, 3*mm))
    story.append(_kv_p("Case ID", spec.header.case_id, st, unicode_ok))
    story.append(_kv_p("Investigator", spec.header.investigator, st, unicode_ok))
    story.append(_kv_p("Station/Unit", spec.header.station_unit, st, unicode_ok))
    story.append(_kv_p("Contact", spec.header.contact, st, unicode_ok))
    if (spec.header.case_notes or "").strip():
        story.append(_kv_p("Case Notes", spec.header.case_notes, st, unicode_ok))
    # QR
    try:
        qr_fit = _resize_fit(Path(qr_path), 400, 400)
        story.append(Spacer(1, 2*mm))
        story.append(RLImage(str(qr_fit), width=35*mm, height=35*mm))
    except Exception:
        pass
    story.append(Spacer(1, 4*mm))

    # Evidence (wrapped, width-safe)
    story.append(Paragraph("Evidence List", st["h2"]))
    headers = ["Filename", "SHA-256", "Ingest time", "Camera ID", "Duration (s)"]
    rows: List[List[Any]] = [[Paragraph(h, st["base"]) for h in headers]]
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

    # Findings
    story.append(Paragraph("Findings", st["h2"]))
    if not spec.findings:
        story.append(Paragraph("No detection/face findings selected.", st["base"]))
    else:
        for i, f in enumerate(spec.findings, 1):
            story.append(Paragraph(f"Event #{i}", st["base"]))
            story.append(_kv_p("Time window", f.time_window, st, unicode_ok))
            story.append(_kv_p("Track ID", "-" if f.track_id is None else str(f.track_id), st, unicode_ok))
            story.append(_kv_p("Object type", f.object_type, st, unicode_ok))
            story.append(_kv_p("BBox (x,y,w,h)", "-" if not f.bbox else str(f.bbox), st, unicode_ok))
            story.append(_kv_p("Matched offender (ID)", f.matched_offender_id or "-", st, unicode_ok))
            story.append(_kv_p("Matched offender (Name)", f.matched_offender_name or "-", st, unicode_ok))
            story.append(_kv_p("Similarity score", "-" if f.similarity_score is None else f"{f.similarity_score:.3f}", st, unicode_ok))
            story.append(_kv_p("Verification status", f.verification_status or "unknown", st, unicode_ok))
            if f.representative_frame_path:
                try:
                    fit = _resize_fit(Path(f.representative_frame_path), 1400, 700)
                    story.append(Spacer(1, 1*mm))
                    story.append(RLImage(str(fit), width=170*mm, height=None))
                except Exception:
                    story.append(Paragraph(f"(Could not render image: {f.representative_frame_path})", st["small"]))
            story.append(Spacer(1, 3*mm))
    story.append(Spacer(1, 3*mm))

    # Forensics (concise, no raw JSON dump)
    story.append(Paragraph("Forensics Summary", st["h2"]))
    md = spec.forensics.metadata_summary or {}
    md_s = _summarize_metadata(md)
    items = [
        f"<b>Path</b>: {md_s.get('path','-')}",
        f"<b>Kind/MIME</b>: {md_s.get('kind','-')} / {md_s.get('mime','-')}",
        f"<b>Duration</b>: {md_s.get('duration','-')}  <b>FPS</b>: {md_s.get('fps','-')}",
        f"<b>Resolution</b>: {md_s.get('res','-')}  <b>Codec</b>: {md_s.get('codec','-')}",
        f"<b>Bitrate</b>: {md_s.get('bitrate','-')}  <b>Size</b>: {md_s.get('size','-')}",
        f"<b>SHA-256 (short)</b>: {md_s.get('sha256','-')}",
    ]
    for line in items:
        story.append(Paragraph(_safe(line, unicode_ok), st["base"]))

    # Tamper flags
    if spec.forensics.tamper_flags:
        story.append(Spacer(1, 1*mm))
        story.append(Paragraph("<b>Tamper checks:</b>", st["base"]))
        for fl in spec.forensics.tamper_flags:
            story.append(Paragraph("• " + _safe(fl, unicode_ok), st["base"]))

    # ELA thumbnails
    if spec.forensics.ela_thumbnails:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph("ELA thumbnails:", st["base"]))
        thumbs: List[RLImage] = []
        for p in spec.forensics.ela_thumbnails[:9]:
            try:
                fit = _resize_fit(Path(p), 600, 300)
                thumbs.append(RLImage(str(fit), width=60*mm, height=None))
            except Exception:
                pass
        for i in range(0, len(thumbs), 3):
            row = thumbs[i:i+3]
            if row:
                t = Table([row], colWidths=[60*mm]*len(row))
                t.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"LEFT"),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
                story.append(t)
                story.append(Spacer(1, 2*mm))

    # Deepfake heuristic
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph("<b>Deepfake heuristic score (experimental):</b>", st["base"]))
    if spec.forensics.deepfake_score is None:
        story.append(Paragraph("N/A", st["base"]))
    else:
        story.append(Paragraph(
            _safe(f"{spec.forensics.deepfake_score:.2f} (0=low suspicion .. 1=high suspicion) - heuristic only; NOT a definitive detector.", unicode_ok),
            st["base"]
        ))
    return story

def _header_footer_cb():
    return (lambda c, d: _header_footer(c, d)), (lambda c, d: _header_footer(c, d))

# ---------------- public API ----------------
def generate_report(spec: ReportSpec) -> Dict[str, Any]:
    """
    Two-pass build:
      Pass 1 -> build body to memory, hash, sign
      Pass 2 -> rebuild fresh with embedded signature page, write final PDF
    """
    _ensure_dirs()

    # IDs / timestamps
    report_id = hashlib.sha1(os.urandom(16)).hexdigest()
    generation_time_utc = datetime.now(timezone.utc).isoformat()

    # JSON bundle
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

    # QR to bundle path (demo)
    qr_path = _qr_png_from_text(str(json_path), f"qr_{report_id}")

    # Fonts/styles
    primary_font, unicode_ok = _setup_fonts()
    st = _styleset(primary_font)

    # ---- PASS 1: build body to memory (no signature page yet)
    spec1 = ReportSpec(
        header=spec.header,
        evidence=spec.evidence,
        findings=spec.findings,
        forensics=spec.forensics,
        bundle_json={"report_id": report_id, "generation_time_utc": generation_time_utc}
    )
    story1 = _build_story(spec1, st, primary_font, unicode_ok, qr_path)
    buf = io.BytesIO()
    doc_mem = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=25*mm, bottomMargin=15*mm,
        title=f"Report {report_id}", author=spec.header.investigator or "unknown",
    )
    onFirst, onLater = _header_footer_cb()
    doc_mem.build(story1, onFirstPage=onFirst, onLaterPages=onLater)
    pdf_bytes = buf.getvalue()
    report_sha256 = _sha256_bytes(pdf_bytes)
    sig_info = _sign_bytes(bytes.fromhex(report_sha256))

    # ---- PASS 2: rebuild fresh body + signature page to disk
    out_pdf = REPORTS_DIR / f"{report_id}.pdf"
    doc = SimpleDocTemplate(
        str(out_pdf), pagesize=A4, leftMargin=15*mm, rightMargin=15*mm, topMargin=25*mm, bottomMargin=15*mm,
        title=f"Report {report_id}", author=spec.header.investigator or "unknown",
    )
    story2 = _build_story(spec1, st, primary_font, unicode_ok, qr_path)
    story2.append(PageBreak())
    story2.append(Paragraph("Signatures", st["h2"]))
    story2.append(Paragraph("The following fields allow integrity verification of this PDF in demo mode.", st["base"]))
    story2.append(Paragraph(f"<b>report_sha256:</b> {_safe(report_sha256, unicode_ok)}", st["mono"]))
    story2.append(Paragraph(f"<b>signature (base64):</b> {_safe(sig_info['signature_b64'][:96] + '…', unicode_ok)}", st["mono"]))
    story2.append(Paragraph(f"<b>signing_cert_subject:</b> {_safe(sig_info['signing_cert_subject'], unicode_ok)}", st["mono"]))
    story2.append(Paragraph(f"<b>signing_cert_pubkey_fingerprint:</b> {_safe(sig_info['signing_cert_pubkey_fingerprint'], unicode_ok)}", st["mono"]))

    onFirst2, onLater2 = _header_footer_cb()
    doc.build(story2, onFirstPage=onFirst2, onLaterPages=onLater2)

    return {
        "report_id": report_id,
        "pdf_path": str(out_pdf),
        "json_path": str(json_path),
        "qr_path": str(qr_path),
        "report_sha256": report_sha256,
        "signature_b64": sig_info["signature_b64"],
        "signing_cert_subject": sig_info["signing_cert_subject"],
        "signing_cert_pubkey_fingerprint": sig_info["signing_cert_pubkey_fingerprint"],
        "generation_time_utc": generation_time_utc,
        "generating_system_version": GENERATING_SYSTEM_VERSION,
    }
