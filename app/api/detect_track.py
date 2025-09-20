# app/api/detect_track.py
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# Local vision modules
from app.vision.detector import YoloDetector, PERSON_AND_VEHICLES
from app.vision.tracker import YoloByteTrack

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
UPLOADS_DIR  = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

router = APIRouter(prefix="/api", tags=["detection-tracking"])

# Default thresholds
DEFAULT_CONF = 0.35
DEFAULT_IOU  = 0.50
DEFAULT_MINLEN = 5   # frames
DEFAULT_STRIDE = 1

def _save_upload(tmp: UploadFile, dst_dir: Path) -> Path:
    suffix = Path(tmp.filename or "upload.bin").suffix
    safe_name = Path(tmp.filename or "upload").name
    out = dst_dir / safe_name
    if out.exists():
        stem = out.stem
        ext = out.suffix
        i = 1
        while True:
            out2 = out.with_name(f"{stem}_{i}{ext}")
            if not out2.exists():
                out = out2
                break
            i += 1
    with out.open("wb") as f:
        shutil.copyfileobj(tmp.file, f)
    return out

def _resolve_uploaded(stored_name: str) -> Path:
    p = (UPLOADS_DIR / stored_name).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Stored file not found: {stored_name}")
    return p

def _rel(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        p = Path(path).resolve()
        return str(p.relative_to(PROJECT_ROOT))
    except Exception:
        return path  # fall back to whatever it is

# ---------------------------
# Upload routes (existing)
# ---------------------------
@router.post("/detect/image")
def detect_image(
    image: UploadFile = File(...),
    model_name: str = Form("yolov8l.pt"),
    conf: float = Form(DEFAULT_CONF),
    iou: float = Form(DEFAULT_IOU),
):
    try:
        path = _save_upload(image, UPLOADS_DIR)
        det = YoloDetector(model_name=model_name, conf=conf, iou=iou, classes=PERSON_AND_VEHICLES)
        res = det.detect_image(path, annotate=True)
        return JSONResponse({
            "ok": True,
            "source_path": _rel(res.source_path),
            "annotated_path": _rel(res.annotated_path),
            "json_path": _rel(res.json_path),
            "num_frames": res.num_frames,
            "total_detections": res.total_detections,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"detect_image failed: {e}")

@router.post("/detect/video")
def detect_video(
    video: UploadFile = File(...),
    model_name: str = Form("yolov8l.pt"),
    conf: float = Form(DEFAULT_CONF),
    iou: float = Form(DEFAULT_IOU),
    stride: int = Form(DEFAULT_STRIDE),
    max_frames: Optional[int] = Form(None),
):
    try:
        path = _save_upload(video, UPLOADS_DIR)
        det = YoloDetector(model_name=model_name, conf=conf, iou=iou, classes=PERSON_AND_VEHICLES)
        res = det.detect_video(path, annotate=True, stride=stride, max_frames=max_frames)
        return JSONResponse({
            "ok": True,
            "source_path": _rel(res.source_path),
            "annotated_path": _rel(res.annotated_path),
            "json_path": _rel(res.json_path),
            "num_frames": res.num_frames,
            "total_detections": res.total_detections,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"detect_video failed: {e}")

@router.post("/track/video")
def track_video(
    video: UploadFile = File(...),
    model_name: str = Form("yolov8l.pt"),
    conf: float = Form(DEFAULT_CONF),
    iou: float = Form(DEFAULT_IOU),
    min_track_len: int = Form(DEFAULT_MINLEN),
    device: Optional[str] = Form(None),
):
    try:
        path = _save_upload(video, UPLOADS_DIR)
        tr = YoloByteTrack(
            model_name=model_name,
            conf=conf,
            iou=iou,
            classes=PERSON_AND_VEHICLES,
            min_track_len=min_track_len,
        )
        res = tr.track_video(path, device=device)
        return JSONResponse({
            "ok": True,
            "source_path": _rel(res.source_path),
            "annotated_path": _rel(res.annotated_path),
            "tracks_json_path": _rel(res.tracks_json_path),
            "findings_json_path": _rel(res.findings_json_path),
            "total_tracks": res.total_tracks,
            "total_frames": res.total_frames,
            "fps": res.fps,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"track_video failed: {e}")

# ---------------------------
# NEW: by-filename routes
# ---------------------------
@router.post("/detect/image/by-filename/{stored_name}")
def detect_image_by_filename(
    stored_name: str,
    model_name: str = Form("yolov8l.pt"),
    conf: float = Form(DEFAULT_CONF),
    iou: float = Form(DEFAULT_IOU),
):
    try:
        path = _resolve_uploaded(stored_name)
        det = YoloDetector(model_name=model_name, conf=conf, iou=iou, classes=PERSON_AND_VEHICLES)
        res = det.detect_image(path, annotate=True)
        return JSONResponse({
            "ok": True,
            "source_path": _rel(res.source_path),
            "annotated_path": _rel(res.annotated_path),
            "json_path": _rel(res.json_path),
            "num_frames": res.num_frames,
            "total_detections": res.total_detections,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"detect_image_by_filename failed: {e}")

@router.post("/detect/video/by-filename/{stored_name}")
def detect_video_by_filename(
    stored_name: str,
    model_name: str = Form("yolov8l.pt"),
    conf: float = Form(DEFAULT_CONF),
    iou: float = Form(DEFAULT_IOU),
    stride: int = Form(DEFAULT_STRIDE),
    max_frames: Optional[int] = Form(None),
):
    try:
        path = _resolve_uploaded(stored_name)
        det = YoloDetector(model_name=model_name, conf=conf, iou=iou, classes=PERSON_AND_VEHICLES)
        res = det.detect_video(path, annotate=True, stride=stride, max_frames=max_frames)
        return JSONResponse({
            "ok": True,
            "source_path": _rel(res.source_path),
            "annotated_path": _rel(res.annotated_path),
            "json_path": _rel(res.json_path),
            "num_frames": res.num_frames,
            "total_detections": res.total_detections,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"detect_video_by_filename failed: {e}")

@router.post("/track/video/by-filename/{stored_name}")
def track_video_by_filename(
    stored_name: str,
    model_name: str = Form("yolov8l.pt"),
    conf: float = Form(DEFAULT_CONF),
    iou: float = Form(DEFAULT_IOU),
    min_track_len: int = Form(DEFAULT_MINLEN),
    device: Optional[str] = Form(None),
):
    try:
        path = _resolve_uploaded(stored_name)
        tr = YoloByteTrack(
            model_name=model_name,
            conf=conf,
            iou=iou,
            classes=PERSON_AND_VEHICLES,
            min_track_len=min_track_len,
        )
        res = tr.track_video(path, device=device)
        return JSONResponse({
            "ok": True,
            "source_path": _rel(res.source_path),
            "annotated_path": _rel(res.annotated_path),
            "tracks_json_path": _rel(res.tracks_json_path),
            "findings_json_path": _rel(res.findings_json_path),
            "total_tracks": res.total_tracks,
            "total_frames": res.total_frames,
            "fps": res.fps,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"track_video_by_filename failed: {e}")
