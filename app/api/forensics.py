# app/api/forensics.py
from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.services.metadata import summarize_forensics
from app.main import PROJECT_ROOT, UPLOAD_DIR  # reuse paths from main

router = APIRouter(prefix="/forensics", tags=["forensics"])


def _resolve_stored_name(stored_name: str) -> Path:
    """
    stored_name is the exact filename under data/uploads (e.g.,
    '9d2d..._sample1.mp4'). We restrict path traversal.
    """
    p = (UPLOAD_DIR / stored_name).resolve()
    if not str(p).startswith(str(UPLOAD_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path.")
    return p


@router.get("/by-name/{stored_name}")
def get_forensics_by_name(stored_name: str):
    """
    Return forensic summary (sha256, mime, image EXIF or video ffprobe, etc.)
    for a previously uploaded file found under data/uploads/.
    """
    path = _resolve_stored_name(stored_name)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        summary = summarize_forensics(path)
        return JSONResponse(summary.as_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forensics error: {e}")
