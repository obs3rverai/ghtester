# app/services/detection_yolo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import os

import cv2
import numpy as np

# Reuse our simple tracker + data structures from the motion stub
from app.services.detection import IOUTracker, Detection as _Detection

# Ultralytics YOLO
_YOLO_MODEL = None  # lazy global


def _get_model(model_name: str = "yolov8n.pt"):
    """
    Lazy-load and cache the YOLO model. The first call will download weights if not present.
    """
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO(model_name)  # e.g., yolov8n.pt
    return _YOLO_MODEL


# COCO class IDs of interest: person(0), car(2), motorcycle(3), bus(5), truck(7)
COCO_KEEP = {0, 2, 3, 5, 7}
ID_TO_LABEL = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    # ... we only use a subset, but labels beyond this map fine
}


def _clip_bbox(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(0, min(x2, W - 1))
    y2 = max(0, min(y2, H - 1))
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return int(x1), int(y1), int(w), int(h)


def analyze_video_with_yolo(
    path: Path,
    sample_fps: float = 2.0,
    max_frames: Optional[int] = 600,
    conf_thresh: float = 0.25,
    iou_thresh_track: float = 0.4
) -> Dict[str, Any]:
    """
    Analyze a video with YOLOv8 detection, and track with our IOU tracker.
    Output shape matches the motion stub:
        { meta, detections:[{frame_idx, ts_sec, bbox, conf, label}], tracks:[...] }
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    stride = max(1, int(round(fps_in / max(0.1, sample_fps))))
    model = _get_model()

    tracker = IOUTracker(iou_thresh=iou_thresh_track, max_misses=10)
    detections: List[_Detection] = []

    frame_idx = -1
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue

        ts_sec = frame_idx / float(max(1e-6, fps_in))

        # YOLO inference (numpy BGR -> model handles conversion)
        results = model.predict(
            source=frame,
            verbose=False,
            conf=conf_thresh,
            imgsz=max(320, int(max(W, H) // 2))  # a small img size keeps it fast
        )

        dets_this_frame: List[_Detection] = []

        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()  # [N,4] x1,y1,x2,y2
            confs = r.boxes.conf.cpu().numpy()  # [N]
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)  # [N]

            for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, cls_ids):
                if cid not in COCO_KEEP:
                    continue
                x, y, w, h = _clip_bbox(int(x1), int(y1), int(x2), int(y2), W, H)
                label = ID_TO_LABEL.get(int(cid), f"id{int(cid)}")
                dets_this_frame.append(_Detection(
                    frame_idx=frame_idx,
                    ts_sec=ts_sec,
                    bbox=(x, y, w, h),
                    conf=float(conf),
                    label=label
                ))

        # Update tracker
        tracker.update(frame_idx, ts_sec, dets_this_frame)
        detections.extend(dets_this_frame)
        processed += 1

        if max_frames is not None and processed >= max_frames:
            break

    cap.release()
    tracker.finalize()

    events = tracker.get_events()
    result = {
        "meta": {
            "video_path": str(path),
            "fps_in": fps_in,
            "fps_sampling": sample_fps,
            "frames_total": total,
            "processed_frames": processed,
            "width": W,
            "height": H,
            "engine": "yolov8",
        },
        "detections": [
            {
                "frame_idx": d.frame_idx,
                "ts_sec": round(d.ts_sec, 3),
                "bbox": [int(d.bbox[0]), int(d.bbox[1]), int(d.bbox[2]), int(d.bbox[3])],
                "conf": round(float(d.conf), 3),
                "label": d.label
            } for d in detections
        ],
        "tracks": [
            {
                "track_id": ev.track_id,
                "label": ev.label,
                "start_frame": ev.start_frame,
                "end_frame": ev.end_frame,
                "start_ts": round(ev.start_ts, 3),
                "end_ts": round(ev.end_ts, 3),
                "best_bbox": [int(ev.best_bbox[0]), int(ev.best_bbox[1]), int(ev.best_bbox[2]), int(ev.best_bbox[3])]
            } for ev in events
        ]
    }
    return result
