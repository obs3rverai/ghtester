# app/main.py
import hashlib
import uuid
from pathlib import Path
from typing import Dict, Optional, Literal, List, Any

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
# FastAPI / Pydantic
from fastapi import Body, Query
from pydantic import BaseModel  # if not already imported
from typing import Optional     # if not already imported

# --- ADD: ELA + Report services ---
from app.services.ela import run_ela_on_image, run_ela_on_video_frame
from app.services.report import (
    EvidenceItem, FindingItem, ForensicBlock, ReportHeader, ReportSpec,
    generate_report, REPORTS_DIR, ASSETS_DIR,
)
# app/main.py  (add these lines)


# --- simple config ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI CCTV & Digital Media Forensic Tool (MVP)", version="0.5.0")

# --- imports from services ---
from app.services.metadata import summarize_forensics, is_video, is_image  # noqa: E402
from app.services.detection import analyze_video_with_motion_stub  # noqa: E402
from app.services.faces import faces_from_image, faces_from_video  # noqa: E402
from app.api.detect_track import router as detect_track_router
app.include_router(detect_track_router)
from app.api.verify import router as verify_router
app.include_router(verify_router)
from app.api.files_admin import router as files_admin_router
app.include_router(files_admin_router)   # exposes DELETE /files/{stored_name}

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

# --- New model for embedding-based search ---
class EmbeddingSearchRequest(BaseModel):
    embedding: List[float]   # L2-normalized ArcFace embedding (usually 512-D)
    top_k: int = 5

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

@app.post("/faces/search/by-embedding")
def faces_search_by_embedding(req: EmbeddingSearchRequest):
    """
    Search the in-memory face index using a precomputed face embedding.
    This bypasses re-detection — ideal for 'Use this detected face as query'.
    """
    if len(_FACE_DB) == 0:
        raise HTTPException(status_code=400, detail="Face index is empty. Add faces first.")

    # ensure numpy vector & normalize for cosine
    q = np.asarray(req.embedding, dtype=np.float32)
    if q.ndim != 1 or q.size == 0:
        raise HTTPException(status_code=400, detail="Invalid embedding shape.")
    q = q / (np.linalg.norm(q) + 1e-12)

    # Stack index embeddings (already L2-normalized during indexing)
    M = np.vstack([item["embedding"] for item in _FACE_DB])  # (N, D)
    sims = (M @ q)  # cosine similarity (N,)

    k = int(max(1, min(int(req.top_k), sims.shape[0])))
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

    return {"top_k": k, "results": results}


@app.post("/faces/search")
async def faces_search(
    file: UploadFile = File(...),
    top_k: int = Form(5)
):
    """
    Upload a query face image (single face recommended). We detect faces,
    take the *largest* face embedding as query, and cosine-search the in-memory index.
    If no face is found, we retry with:
      1) lenient detector settings
      2) padded + enhanced image + lenient detector
    """
    if len(_FACE_DB) == 0:
        raise HTTPException(status_code=400, detail="Face index is empty. Add faces first via /faces/index/add/by-filename/...")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")

    # Save to uploads (debug)
    media_id = str(uuid.uuid4())
    safe_name = file.filename.replace("/", "_").replace("\\", "_") or "query.jpg"
    tmp_path = UPLOAD_DIR / f"{media_id}_{safe_name}"
    with open(tmp_path, "wb") as out:
        out.write(raw)

    # Must be an image
    if not is_image(tmp_path):
        raise HTTPException(status_code=400, detail="Query must be an image file.")

    # ---------- Try 1: normal pipeline ----------
    qres = faces_from_image(tmp_path)
    faces = qres.get("faces", [])

    # ---------- Try 2: lenient pipeline ----------
    if not faces:
        try:
            from app.services.faces import FaceEngine
            fe_lenient = FaceEngine(
                providers=None,
                det_size=1280,
                min_face_size=10,
                min_sharpness=0.0,
                pre_upscale_if_below_height=4096,
                pre_upscale_factor=2.0
            )
            qres2 = fe_lenient.extract_from_image(tmp_path)
            faces = qres2.get("faces", [])
        except Exception:
            faces = []

    # ---------- Try 3: pad + enhance + lenient pipeline ----------
    if not faces:
        try:
            import cv2
            import numpy as np
            from app.services.faces import FaceEngine

            img = cv2.imread(str(tmp_path))
            if img is None or img.size == 0:
                raise RuntimeError("Could not read query image.")

            H, W = img.shape[:2]
            # a) Unsharp mask (light)
            blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
            sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

            # b) CLAHE on L-channel (LAB)
            lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            L2 = clahe.apply(L)
            lab2 = cv2.merge([L2, A, B])
            enh = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

            # c) Add 30% replicate padding around (helps very tight crops)
            top = bottom = int(0.30 * H)
            left = right = int(0.30 * W)
            padded = cv2.copyMakeBorder(enh, top, bottom, left, right, cv2.BORDER_REPLICATE)

            # d) Upscale small inputs
            short_side = min(padded.shape[:2])
            if short_side < 400:
                scale = 400.0 / max(1.0, short_side)
                padded = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # write a temp padded copy
            padded_path = UPLOAD_DIR / f"{media_id}_padded_{safe_name}"
            cv2.imwrite(str(padded_path), padded)

            fe_pad = FaceEngine(
                providers=None,
                det_size=1280,
                min_face_size=8,
                min_sharpness=0.0,
                pre_upscale_if_below_height=4096,
                pre_upscale_factor=2.0
            )
            qres3 = fe_pad.extract_from_image(padded_path)
            faces_pad = qres3.get("faces", [])

            # If we detected on the padded image, just use the largest face's embedding
            if faces_pad:
                faces = faces_pad
        except Exception:
            pass

    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in query image (after padded+enhanced retry).")

    # pick largest face by area
    def area(bb):
        x, y, w, h = bb
        return int(w) * int(h)
    faces_sorted = sorted(faces, key=lambda f: area(f["bbox"]), reverse=True)

    # Embedding
    q_emb = _emb_from_face_dict(faces_sorted[0])

    # Cosine search
    M = np.vstack([item["embedding"] for item in _FACE_DB])  # (N,D)
    qn = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    sims = (M @ qn)

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

# ---------- Forensics: ELA (image or single video frame) ----------
@app.post("/forensics/ela/by-filename/{stored_name}")
def forensics_ela_by_filename(
    stored_name: str,
    jpeg_quality: int = Query(90, ge=10, le=95),
    hi_thresh: int = Query(40, ge=0, le=255),
    frame_idx: Optional[int] = Query(None, ge=0),
):
    """
    Run Error Level Analysis (ELA).
    - Images: run directly.
    - Videos: provide frame_idx to extract that frame first.
    Returns ELA stats and the path to the ELA PNG under data/reports/assets/.
    """
    src_path = (UPLOAD_DIR / stored_name).resolve()
    if not src_path.exists():
        raise HTTPException(status_code=404, detail=f"Stored file not found: {stored_name}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    suffix = src_path.suffix.lower()
    is_image = suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    is_video = suffix in {".mp4", ".mov", ".mkv", ".avi"}

    try:
        if is_image:
            res = run_ela_on_image(src_path, out_dir=ASSETS_DIR, jpeg_quality=int(jpeg_quality), hi_thresh=int(hi_thresh))
        elif is_video:
            if frame_idx is None:
                raise HTTPException(status_code=400, detail="For videos, provide frame_idx.")
            res = run_ela_on_video_frame(src_path, frame_idx=int(frame_idx), out_dir=ASSETS_DIR,
                                         jpeg_quality=int(jpeg_quality), hi_thresh=int(hi_thresh))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type for ELA.")

        # Return paths relative to project root for the UI
        def rel(p: str) -> str:
            return str(Path(p).resolve().relative_to(PROJECT_ROOT))

        out = res.to_dict()
        out["original_path"] = rel(out["original_path"])
        out["ela_path"] = rel(out["ela_path"])
        return out

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ELA failed: {e}")


# ---------- Report: Generate PDF ----------
class ReportGenerateRequest(BaseModel):
    header: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    findings: List[Dict[str, Any]]
    forensics: Dict[str, Any]
    bundle_json: Optional[Dict[str, Any]] = None

@app.post("/report/generate")
def report_generate(req: ReportGenerateRequest):
    """
    Build a PDF that includes:
      - Header (Case ID, Investigator, Station/Unit, Contact, Case Notes)
      - Evidence table
      - Findings with representative images
      - Forensics (metadata summary, tamper flags, ELA thumbnails, deepfake score)
      - Signatures (SHA-256 + demo self-signed signature)
    Returns paths and signature details.
    """
    try:
        # ---- Build dataclasses from request ----
        hdr = req.header
        header = ReportHeader(
            case_id=str(hdr.get("case_id", "")),
            investigator=str(hdr.get("investigator", "")),
            station_unit=str(hdr.get("station_unit", "")),
            contact=str(hdr.get("contact", "")),
            case_notes=str(hdr.get("case_notes", "")),
        )

        evidence_items: List[EvidenceItem] = []
        for e in req.evidence:
            evidence_items.append(EvidenceItem(
                filename=str(e.get("filename", "")),
                sha256=str(e.get("sha256", "")),
                ingest_time=str(e.get("ingest_time", "")),
                camera_id=str(e.get("camera_id", "unknown")),
                duration=(float(e["duration"]) if e.get("duration") is not None else None),
            ))

        findings_items: List[FindingItem] = []
        for f in req.findings:
            bbox_val = f.get("bbox")
            bbox_t = tuple(bbox_val) if bbox_val is not None else None
            findings_items.append(FindingItem(
                time_window=str(f.get("time_window", "")),
                track_id=(int(f["track_id"]) if f.get("track_id") is not None else None),
                object_type=str(f.get("object_type", "")),
                representative_frame_path=str(f.get("representative_frame_path")) if f.get("representative_frame_path") else None,
                bbox=bbox_t,  # (x,y,w,h)
                matched_offender_id=str(f.get("matched_offender_id")) if f.get("matched_offender_id") else None,
                matched_offender_name=str(f.get("matched_offender_name")) if f.get("matched_offender_name") else None,
                similarity_score=(float(f["similarity_score"]) if f.get("similarity_score") is not None else None),
                verification_status=str(f.get("verification_status", "unknown")),
            ))

        fx = req.forensics
        fb = ForensicBlock(
            metadata_summary=dict(fx.get("metadata_summary", {})),
            tamper_flags=[str(x) for x in fx.get("tamper_flags", [])],
            ela_thumbnails=[str(x) for x in fx.get("ela_thumbnails", [])],
            deepfake_score=(float(fx["deepfake_score"]) if fx.get("deepfake_score") is not None else None),
        )

        spec = ReportSpec(
            header=header,
            evidence=evidence_items,
            findings=findings_items,
            forensics=fb,
            bundle_json=req.bundle_json or {},
        )

        # ---- Generate PDF ----
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        result = generate_report(spec)

        # Return paths relative to project root for the UI
        def rel(p: str) -> str:
            return str(Path(p).resolve().relative_to(PROJECT_ROOT))

        return {
            **result,
            "pdf_path": rel(result["pdf_path"]),
            "json_path": rel(result["json_path"]),
            "qr_path": rel(result["qr_path"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")
