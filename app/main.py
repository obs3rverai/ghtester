# app/main.py
import hashlib
import uuid
from pathlib import Path
from typing import Dict, Optional, Literal, List

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# --- simple config ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI CCTV & Digital Media Forensic Tool (MVP)", version="0.5.0")

# --- imports from services ---
from app.services.metadata import summarize_forensics, is_video, is_image  # noqa: E402
from app.services.detection import analyze_video_with_motion_stub  # noqa: E402
from app.services.faces import faces_from_image, faces_from_video  # noqa: E402

# Try to import YOLO service (optional)
try:
    from app.services.detection_yolo import analyze_video_with_yolo  # noqa: E402
    _YOLO_AVAILABLE = True
except Exception:
    analyze_video_with_yolo = None  # type: ignore
    _YOLO_AVAILABLE = False


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


class DetectParams(BaseModel):
    # shared
    engine: Literal["motion", "yolov8"] = Field("motion", description="Choose detection engine")
    sample_fps: float = Field(2.0, ge=0.1, le=60.0, description="Frames processed per second")
    max_frames: Optional[int] = Field(600, ge=1, description="Limit frames for speed; null=all")

    # motion-stub specific
    min_area: int = Field(500, ge=10, description="(motion) Min contour area")

    # YOLO-specific
    conf_thresh: float = Field(0.25, ge=0.01, le=0.95, description="(yolov8) Confidence threshold")
    iou_thresh_track: float = Field(0.4, ge=0.1, le=0.9, description="(yolov8) IOU for track association")

    @validator("max_frames", pre=True)
    def _none_ok(cls, v):
        # allow null/None from clients
        return v


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
# Basic endpoints
# =========================

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "yolo_available": "yes" if _YOLO_AVAILABLE else "no"}


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
# Forensics endpoints
# =========================

@app.get("/forensics/files")
def forensics_files():
    files = [p.name for p in sorted(UPLOAD_DIR.glob("*")) if p.is_file()]
    return {"count": len(files), "files": files}


@app.get("/forensics/summary/by-filename/{stored_name}", response_model=ForensicResponse)
def forensics_by_filename(stored_name: str):
    path = _safe_upload_path(stored_name)
    summary = summarize_forensics(path).as_dict()
    return JSONResponse(summary)


@app.get("/forensics/summary/by-media-id/{media_id}", response_model=ForensicResponse)
def forensics_by_media_id(media_id: str):
    path = _find_by_media_id(media_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"No file found for media_id={media_id}")
    summary = summarize_forensics(path).as_dict()
    return JSONResponse(summary)


# =========================
# Detection endpoints (engine switch)
# =========================

@app.get("/detect/options")
def detect_options():
    """Default parameters to guide the UI."""
    return DetectParams().dict()

def _run_detection(path: Path, params: DetectParams) -> Dict:
    if not is_video(path):
        raise HTTPException(status_code=400, detail="Selected file is not recognized as a video.")

    if params.engine == "motion":
        return analyze_video_with_motion_stub(
            path=path,
            sample_fps=params.sample_fps,
            max_frames=params.max_frames,
            min_area=params.min_area,
        )
    elif params.engine == "yolov8":
        if not _YOLO_AVAILABLE or analyze_video_with_yolo is None:
            raise HTTPException(
                status_code=503,
                detail="YOLOv8 engine not available. Install ultralytics + torch and restart the server."
            )
        return analyze_video_with_yolo(
            path=path,
            sample_fps=params.sample_fps,
            max_frames=params.max_frames,
            conf_thresh=params.conf_thresh,
            iou_thresh_track=params.iou_thresh_track,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown engine: {params.engine}")

@app.post("/detect/run/by-filename/{stored_name}")
def detect_by_filename(stored_name: str, params: DetectParams):
    path = _safe_upload_path(stored_name)
    result = _run_detection(path, params)
    return JSONResponse(result)

@app.post("/detect/run/by-media-id/{media_id}")
def detect_by_media_id(media_id: str, params: DetectParams):
    path = _find_by_media_id(media_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"No file found for media_id={media_id}")
    result = _run_detection(path, params)
    return JSONResponse(result)


# =========================
# Faces: extract, index, search (NEW)
# =========================

class FaceExtractParams(BaseModel):
    sample_fps: float = Field(1.0, ge=0.1, le=30.0, description="Video sampling FPS (images ignore)")
    max_frames: Optional[int] = Field(200, ge=1, description="Limit sampled frames for video; null=all")

@app.post("/faces/extract/by-filename/{stored_name}")
def faces_extract_by_filename(stored_name: str, params: FaceExtractParams):
    """
    If image: returns faces + embeddings from the image.
    If video: samples frames at sample_fps and returns faces + embeddings (with frame_idx, ts_sec).
    """
    path = _safe_upload_path(stored_name)
    if is_image(path):
        out = faces_from_image(path)
    elif is_video(path):
        out = faces_from_video(path, sample_fps=params.sample_fps, max_frames=params.max_frames)
    else:
        raise HTTPException(status_code=400, detail="File is neither image nor video.")
    return JSONResponse(out)

# --- simple in-memory face index (reset on server restart) ---
_FACE_DB: List[Dict] = []    # each item: { "embedding": np.ndarray, "source_file": str, "bbox": [x,y,w,h], "frame_idx": int|None, "ts_sec": float|None }

def _emb_from_face_dict(face_d: Dict) -> np.ndarray:
    emb = np.asarray(face_d.get("normed_embedding", []), dtype=np.float32)
    if emb.ndim != 1 or emb.size == 0:
        raise ValueError("Invalid embedding in face object")
    # ensure L2 normalized (safety)
    n = np.linalg.norm(emb) + 1e-12
    return (emb / n).astype(np.float32)

@app.post("/faces/index/reset")
def faces_index_reset():
    _FACE_DB.clear()
    return {"status": "ok", "count": 0}

class FaceIndexAddParams(BaseModel):
    sample_fps: float = Field(1.0, ge=0.1, le=30.0)
    max_frames: Optional[int] = Field(100, ge=1)
    max_faces: int = Field(1000, ge=1, le=10000)

@app.post("/faces/index/add/by-filename/{stored_name}")
def faces_index_add_by_filename(stored_name: str, params: FaceIndexAddParams):
    """
    Extract faces from the given stored file and add them to the in-memory index.
    """
    path = _safe_upload_path(stored_name)
    if is_image(path):
        out = faces_from_image(path)
        faces = out.get("faces", [])
    elif is_video(path):
        out = faces_from_video(path, sample_fps=params.sample_fps, max_frames=params.max_frames)
        faces = out.get("faces", [])
    else:
        raise HTTPException(status_code=400, detail="File is neither image nor video.")

    added = 0
    for f in faces[: params.max_faces]:
        try:
            emb = _emb_from_face_dict(f)
        except Exception:
            continue
        _FACE_DB.append({
            "embedding": emb,
            "source_file": stored_name,
            "bbox": f.get("bbox"),
            "frame_idx": f.get("frame_idx"),
            "ts_sec": f.get("ts_sec"),
        })
        added += 1

    return {"status": "ok", "added": added, "total": len(_FACE_DB)}

@app.get("/faces/index/stats")
def faces_index_stats():
    return {"count": len(_FACE_DB)}

def _cosine_sim_matrix(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    # q: (D,), M: (N,D)
    qn = q / (np.linalg.norm(q) + 1e-12)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return (Mn @ qn)

@app.post("/faces/search")
async def faces_search(
    file: UploadFile = File(...),
    top_k: int = Form(5)
):
    """
    Upload a query face image (single face recommended). We detect faces,
    take the *largest* face embedding as query, and cosine-search the in-memory index.
    If no face is found at first, we retry with a more lenient detector configuration.
    """
    if len(_FACE_DB) == 0:
        raise HTTPException(status_code=400, detail="Face index is empty. Add faces first via /faces/index/add/by-filename/...")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")

    # Save to a temp path inside uploads (optional for debugging)
    media_id = str(uuid.uuid4())
    safe_name = file.filename.replace("/", "_").replace("\\", "_") or "query.jpg"
    tmp_path = UPLOAD_DIR / f"{media_id}_{safe_name}"
    with open(tmp_path, "wb") as out:
        out.write(raw)

    # Query must be an image file
    if not is_image(tmp_path):
        raise HTTPException(status_code=400, detail="Query must be an image file.")

    # --- Try 1: normal settings (strict) ---
    qres = faces_from_image(tmp_path)
    faces = qres.get("faces", [])

    # --- Try 2: lenient settings (bigger det_size, allow tiny/soft faces, strong upscale) ---
    if not faces:
        try:
            from app.services.faces import FaceEngine  # use same implementation with relaxed thresholds
            fe_lenient = FaceEngine(
                providers=None,
                det_size=1280,
                min_face_size=10,         # accept small crops
                min_sharpness=0.0,        # accept soft crops
                pre_upscale_if_below_height=4096,
                pre_upscale_factor=2.0
            )
            qres2 = fe_lenient.extract_from_image(tmp_path)
            faces = qres2.get("faces", [])
        except Exception:
            faces = []

    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in query image (even after lenient retry).")

    # pick largest bbox face
    def area(bb):
        x, y, w, h = bb
        return int(w) * int(h)
    faces_sorted = sorted(faces, key=lambda f: area(f["bbox"]), reverse=True)
    q_emb = _emb_from_face_dict(faces_sorted[0])

    # Prepare matrix & cosine sim
    M = np.vstack([item["embedding"] for item in _FACE_DB])  # (N,D)
    sims = _cosine_sim_matrix(q_emb, M)  # (N,)

    # Top-k
    k = int(max(1, min(top_k, sims.shape[0])))
    top_idx = np.argsort(-sims)[:k]

    results = []
    for i in top_idx:
        item = _FACE_DB[int(i)]
        results.append({
            "score": float(sims[int(i)]),
            "source_file": item["source_file"],
            "bbox": item["bbox"],
            "frame_idx": item["frame_idx"],
            "ts_sec": item["ts_sec"],
        })

    return {
        "query_saved_path": str(tmp_path.relative_to(PROJECT_ROOT)),
        "query_faces_found": len(faces),
        "top_k": k,
        "results": results
    }
