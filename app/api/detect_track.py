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

# Default thresholds (balanced for accuracy on CPU)
DEFAULT_CONF = 0.35
DEFAULT_IOU  = 0.50
DEFAULT_MINLEN = 5   # frames
DEFAULT_STRIDE = 1   # process each frame; set 2+ for speed on big videos

def _save_upload(tmp: UploadFile, dst_dir: Path) -> Path:
    suffix = Path(tmp.filename or "upload.bin").suffix
    safe_name = Path(tmp.filename or "upload").name
    out = dst_dir / safe_name
    # If a file with same name exists, append a numeric suffix
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

# ---------------------------
# /api/detect/image
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
            "source_path": res.source_path,
            "annotated_path": res.annotated_path,
            "json_path": res.json_path,
            "num_frames": res.num_frames,
            "total_detections": res.total_detections,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"detect_image failed: {e}")

# ---------------------------
# /api/detect/video
# ---------------------------
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
            "source_path": res.source_path,
            "annotated_path": res.annotated_path,
            "json_path": res.json_path,
            "num_frames": res.num_frames,
            "total_detections": res.total_detections,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"detect_video failed: {e}")

# ---------------------------
# /api/track/video  (YOLO + ByteTrack)
# ---------------------------
@router.post("/track/video")
def track_video(
    video: UploadFile = File(...),
    model_name: str = Form("yolov8l.pt"),
    conf: float = Form(DEFAULT_CONF),
    iou: float = Form(DEFAULT_IOU),
    min_track_len: int = Form(DEFAULT_MINLEN),
    device: Optional[str] = Form(None),  # "cpu" or "0" for first GPU
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
            "source_path": res.source_path,
            "annotated_path": res.annotated_path,
            "tracks_json_path": res.tracks_json_path,
            "findings_json_path": res.findings_json_path,
            "total_tracks": res.total_tracks,
            "total_frames": res.total_frames,
            "fps": res.fps,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"track_video failed: {e}")
