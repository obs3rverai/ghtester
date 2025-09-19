# app/services/detection.py
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
from pathlib import Path


# -------------------------
# Data models (plain Python)
# -------------------------

@dataclass
class Detection:
    frame_idx: int
    ts_sec: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    conf: float
    label: str  # e.g., "motion" (stub) | later "person"/"car"

@dataclass
class Track:
    id: int
    label: str
    # running state
    bbox: Tuple[int, int, int, int]
    last_frame_idx: int
    last_ts_sec: float
    hits: int = 0
    misses: int = 0

@dataclass
class TrackEvent:
    track_id: int
    label: str
    start_frame: int
    end_frame: int
    start_ts: float
    end_ts: float
    best_bbox: Tuple[int, int, int, int]
    total_hits: int


# -------------------------
# Utility functions
# -------------------------

def iou_xywh(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / float(union + 1e-9)


def to_bbox(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
    return (int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1)))


# -------------------------
# Naive Motion Detector
# -------------------------

class NaiveMotionDetector:
    """
    Very fast, zero-dependency motion-based detector.
    Produces 'motion' bboxes by frame differencing + contour extraction.
    This is ONLY for wiring & demo; will be replaced by YOLO later.
    """
    def __init__(self, min_area: int = 500, dilate_iters: int = 2):
        self.prev = None
        self.min_area = min_area
        self.dilate_iters = dilate_iters

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], float, str]]:
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (5,5), 0)

        if self.prev is None:
            self.prev = frame_gray
            return []

        # frame differencing
        delta = cv2.absdiff(self.prev, frame_gray)
        self.prev = frame_gray

        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=self.dilate_iters)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            # confidence is a heuristic from area
            conf = min(0.99, 0.5 + math.log10(max(10.0, area)) / 5.0)
            dets.append((to_bbox(x, y, x + w, y + h), conf, "motion"))
        return dets


# -------------------------
# Tiny IOU Tracker (single class)
# -------------------------

class IOUTracker:
    """
    Dead-simple tracker: associates current detections to existing tracks using IOU.
    Creates new tracks for unmatched detections; aged out after max_misses.
    """
    def __init__(self, iou_thresh: float = 0.3, max_misses: int = 10):
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.events_open: Dict[int, TrackEvent] = {}
        self.events_closed: List[TrackEvent] = []

    def update(self, frame_idx: int, ts_sec: float, detections: List[Detection]) -> None:
        # Step 1: Build lists for assignment
        det_bboxes = [d.bbox for d in detections]
        det_labels  = [d.label for d in detections]
        det_used = [False]*len(detections)

        # Step 2: Try to match existing tracks by IOU
        for tid, tr in list(self.tracks.items()):
            # pick best matching detection for this track
            best_iou, best_k = 0.0, -1
            for k, (bb) in enumerate(det_bboxes):
                if det_used[k]:
                    continue
                i = iou_xywh(tr.bbox, bb)
                if i > best_iou:
                    best_iou, best_k = i, k

            if best_k >= 0 and best_iou >= self.iou_thresh:
                # assign
                tr.bbox = det_bboxes[best_k]
                tr.last_frame_idx = frame_idx
                tr.last_ts_sec = ts_sec
                tr.hits += 1
                tr.misses = 0
                det_used[best_k] = True

                # update event window
                ev = self.events_open.get(tid)
                if ev:
                    ev.end_frame = frame_idx
                    ev.end_ts = ts_sec
                    # naive "best bbox" = latest bbox; could be lengthiest/higher area
                    ev.best_bbox = tr.bbox
            else:
                # no assignment
                tr.misses += 1
                if tr.misses > self.max_misses:
                    # close event if open
                    ev = self.events_open.pop(tid, None)
                    if ev:
                        self.events_closed.append(ev)
                    del self.tracks[tid]

        # Step 3: Create new tracks for unmatched detections
        for k, used in enumerate(det_used):
            if used:
                continue
            bbox = det_bboxes[k]
            label = det_labels[k]
            tid = self.next_id
            self.next_id += 1
            tr = Track(id=tid, label=label, bbox=bbox, last_frame_idx=frame_idx, last_ts_sec=ts_sec, hits=1, misses=0)
            self.tracks[tid] = tr
            # open event
            self.events_open[tid] = TrackEvent(
                track_id=tid, label=label,
                start_frame=frame_idx, end_frame=frame_idx,
                start_ts=ts_sec, end_ts=ts_sec,
                best_bbox=bbox, total_hits=1
            )

    def finalize(self):
        # move open events to closed at end
        for tid, ev in list(self.events_open.items()):
            self.events_closed.append(ev)
        self.events_open.clear()

    def get_events(self) -> List[TrackEvent]:
        return list(self.events_closed)


# -------------------------
# Public API
# -------------------------

def analyze_video_with_motion_stub(
    path: Path,
    sample_fps: float = 2.0,
    max_frames: Optional[int] = 600,
    min_area: int = 500
) -> Dict[str, Any]:
    """
    Analyze a video using a fast motion-based detector and an IOU tracker.
    Returns a JSON-serializable dict with per-frame detections and track events.

    Args:
        path: path to video file
        sample_fps: frames processed per second (use <= video fps)
        max_frames: limit frames for speed; None to process all
        min_area: minimum contour area to create a detection

    Output dict keys:
        - meta: {fps_in, fps_sampling, frames_total, processed_frames, width, height}
        - detections: [ {frame_idx, ts_sec, bbox:[x,y,w,h], conf, label}, ... ]
        - tracks: [ {track_id, label, start_frame, end_frame, start_ts, end_ts, best_bbox:[...] }, ... ]
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # sampling stride
    stride = max(1, int(round(fps_in / max(0.1, sample_fps))))

    detector = NaiveMotionDetector(min_area=min_area)
    tracker = IOUTracker(iou_thresh=0.3, max_misses=10)

    detections: List[Detection] = []

    frame_idx = -1
    processed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Sampling
        if frame_idx % stride != 0:
            continue

        ts_sec = frame_idx / float(max(1e-6, fps_in))
        # Detect motion bboxes
        det_tuples = detector.detect(frame)  # list of (bbox, conf, label)
        dets = [Detection(frame_idx, ts_sec, bb, conf, lbl) for (bb, conf, lbl) in det_tuples]

        # Track
        tracker.update(frame_idx, ts_sec, dets)

        detections.extend(dets)
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
            "width": width,
            "height": height,
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
