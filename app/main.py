# app/main.py
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- simple config ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI CCTV & Digital Media Forensic Tool (MVP)", version="0.2.0")

# --- imports from services ---
from app.services.metadata import summarize_forensics  # noqa: E402


# =========================
# Models
# =========================

class IngestResponse(BaseModel):
    media_id: str
    filename: str
    size_bytes: int
    sha256: str
    stored_path: str


class ForensicResponse(BaseModel):
    path: str
    size_bytes: int
    sha256: str
    mime: str
    kind: str
    details: dict


# =========================
# Helpers
# =========================

def _safe_upload_path(name: str) -> Path:
    """
    Return a safe path inside UPLOAD_DIR for a stored filename.
    Reject path traversal.
    """
    p = (UPLOAD_DIR / name).resolve()
    if UPLOAD_DIR.resolve() not in p.parents:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return p


def _find_by_media_id(media_id: str) -> Optional[Path]:
    """
    Find first file matching '<media_id>_*' inside UPLOAD_DIR.
    """
    for p in UPLOAD_DIR.glob(f"{media_id}_*"):
        if p.is_file():
            return p
    return None


# =========================
# Basic endpoints (from Step 2)
# =========================

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Accept an uploaded media file (image or video), compute SHA-256,
    and store it under data/uploads/<media_id>_<sanitized_name>.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    media_id = str(uuid.uuid4())
    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    dest_path = UPLOAD_DIR / f"{media_id}_{safe_name}"

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")

    sha = hashlib.sha256(raw).hexdigest()
    with open(dest_path, "wb") as out:
        out.write(raw)

    resp = IngestResponse(
        media_id=media_id,
        filename=safe_name,
        size_bytes=len(raw),
        sha256=sha,
        stored_path=str(dest_path.relative_to(PROJECT_ROOT)),
    )
    return JSONResponse(resp.dict())


@app.get("/files")
def list_files():
    """
    List stored files (media_id is part of the filename).
    """
    files = []
    for p in sorted(UPLOAD_DIR.glob("*")):
        if p.is_file():
            files.append(p.name)
    return {"count": len(files), "files": files}


# =========================
# Forensics endpoints (NEW)
# =========================

@app.get("/forensics/files")
def forensics_files():
    """
    Alias to list files (handy for the UI).
    """
    files = [p.name for p in sorted(UPLOAD_DIR.glob("*")) if p.is_file()]
    return {"count": len(files), "files": files}


@app.get("/forensics/summary/by-filename/{stored_name}", response_model=ForensicResponse)
def forensics_by_filename(stored_name: str):
    """
    Return forensic summary for a given stored filename in data/uploads.
    Example stored_name: '123e4567-..._sample.jpg'
    """
    path = _safe_upload_path(stored_name)
    summary = summarize_forensics(path).as_dict()
    return JSONResponse(summary)


@app.get("/forensics/summary/by-media-id/{media_id}", response_model=ForensicResponse)
def forensics_by_media_id(media_id: str):
    """
    Return forensic summary for the first file whose name starts with <media_id>_.
    """
    path = _find_by_media_id(media_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"No file found for media_id={media_id}")
    summary = summarize_forensics(path).as_dict()
    return JSONResponse(summary)
