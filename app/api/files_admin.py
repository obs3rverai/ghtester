# app/api/files_admin.py
from __future__ import annotations

from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
UPLOADS_DIR  = DATA_DIR / "uploads"
ANNOT_DIR    = DATA_DIR / "annotated"
DET_DIR      = DATA_DIR / "detections"
TRK_DIR      = DATA_DIR / "tracks"
FIND_DIR     = DATA_DIR / "findings"

for p in (UPLOADS_DIR, ANNOT_DIR, DET_DIR, TRK_DIR, FIND_DIR):
    p.mkdir(parents=True, exist_ok=True)

router = APIRouter(tags=["files-admin"])

def _safe_name(name: str) -> str:
    # prevent path traversal
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return name

def _delete_related(stem: str) -> list[str]:
    deleted: List[str] = []
    for folder in (ANNOT_DIR, DET_DIR, TRK_DIR, FIND_DIR):
        for p in folder.glob(stem + ".*"):
            try:
                p.unlink(missing_ok=True)
                deleted.append(str(p.relative_to(PROJECT_ROOT)))
            except Exception:
                # ignore single-file errors; continue
                pass
    return deleted

@router.delete("/files/{stored_name}")
def delete_uploaded_file(
    stored_name: str,
    deep: bool = Query(False, description="Also delete related artifacts (annotated/detections/tracks/findings)"),
):
    name = _safe_name(stored_name)
    path = (UPLOADS_DIR / name).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    try:
        path.unlink()
    except Exception as e:
        raise HTTPException(status_code=409, detail=f"Could not delete (in use?): {e}")

    related = []
    if deep:
        related = _delete_related(path.stem)

    return {
        "ok": True,
        "deleted": str(path.relative_to(PROJECT_ROOT)),
        "related_deleted": related,
    }
