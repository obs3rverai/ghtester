# app/main.py
import os
import hashlib
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- simple config ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI CCTV & Digital Media Forensic Tool (MVP)", version="0.1.0")


class IngestResponse(BaseModel):
    media_id: str
    filename: str
    size_bytes: int
    sha256: str
    stored_path: str


def sha256_filelike(fobj, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute SHA-256 hash of a file-like object by streaming chunks.
    The file pointer will be consumed.
    """
    h = hashlib.sha256()
    total = 0
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            break
        h.update(chunk)
        total += len(chunk)
    return h.hexdigest()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Accept an uploaded media file (image or video), compute SHA-256,
    and store it under data/uploads/<media_id>_<sanitized_name>.
    """
    # Basic validations
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    # Generate media_id and storage path
    media_id = str(uuid.uuid4())
    safe_name = file.filename.replace("/", "_").replace("\\", "_")
    dest_path = UPLOAD_DIR / f"{media_id}_{safe_name}"

    # Read into memory once to compute hash and write (simple for MVP)
    # For large files, switch to chunked streaming on disk, but this is fine for hackathon samples.
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
