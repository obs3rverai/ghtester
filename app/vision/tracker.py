# app/vision/tracker.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Reuse class set from detector (copy constants to keep this file standalone)
COCO_NAMES = (
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
    "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
)
PERSON_AND_VEHICLES = {0, 1, 2, 3, 5, 7}  # person + (bicycle, car, motorcycle, bus, truck)

# ---------- paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
IN_DIR       = DATA_DIR / "uploads"
OUT_DIR      = DATA_DIR / "tracks"
FRAMES_DIR   = DATA_DIR / "frames" / "tracks"    # separate sub-folder per your request
ANNOT_DIR    = DATA_DIR / "annotated"            # reuse annotated root

for p in [OUT_DIR, FRAMES_DIR, ANNOT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ---------- dataclasses ----------
@dataclass
class TrackPoint:
    frame_index: int
    time_sec: float
    xyxy: Tuple[int, int, int, int]

@dataclass
class TrackItem:
    track_id: int
    cls_id: int
    cls_name: str
    conf_avg: float
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    points: List[TrackPoint]
    repr_frame_path: Optional[str]

@dataclass
class TrackingResult:
    source_path: str
    annotated_path: Optional[str]
    tracks_json_path: str
    findings_json_path: str
    total_tracks: int
    total_frames: int
    fps: float

# ---------- helpers ----------
def _color_for_id(tid: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed=tid * 7919)
    return tuple(int(x) for x in rng.integers(50, 230, 3))  # BGR

def _midpoint_idx(start: int, end: int) -> int:
    return start + (end - start) // 2

def _save_frame(video_path: str, frame_index: int, out_path: Path) -> Optional[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return str(out_path)

def _sec_to_window(s1: float, s2: float) -> str:
    # “start..end (s)” string; the report generator just shows what we pass
    return f"{s1:.2f}s .. {s2:.2f}s"

# ---------- core tracker ----------
class YoloByteTrack:
    """
    Ultralytics YOLO + ByteTrack via model.track()
    Produces:
      - annotated MP4 with ID labels
      - structured tracks.json
      - derived findings.json compatible with your report 'Findings' section
    """
    def __init__(
        self,
        model_name: str = "yolov8l.pt",
        conf: float = 0.35,
        iou: float = 0.50,
        classes: Optional[set[int]] = None,
        tracker_yaml: str = "bytetrack.yaml",
        min_track_len: int = 5,      # frames
    ):
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self.classes = PERSON_AND_VEHICLES if classes is None else classes
        self.tracker_yaml = tracker_yaml
        self.min_track_len = min_track_len

    def track_video(
        self,
        video_path: str | Path,
        out_name: Optional[str] = None,
        save_annotated: bool = True,
        device: Optional[str] = None,   # e.g. "cpu" or "0"
    ) -> TrackingResult:
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        # Where to save annotated MP4
        annotated_path = None
        if save_annotated:
            out_name = out_name or (Path(video_path).stem + ".track.mp4")
            annotated_path = str(ANNOT_DIR / out_name)

        # Run ultralytics tracker (saves an annotated file in runs/ by default; we’ll also draw our own)
        # We use stream=True to iterate and draw with consistent styling and ID labels.
        stream = self.model.track(
            source=video_path,
            conf=self.conf,
            iou=self.iou,
            classes=sorted(self.classes),
            tracker=self.tracker_yaml,
            stream=True,
            verbose=False,
            device=device if device else None,
            persist=True,  # keep track IDs across frames
        )

        # Pre-create writer if annotating
        writer = None
        if save_annotated:
            # fetch width/height from first frame lazily
            cap = cv2.VideoCapture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(annotated_path, fourcc, fps, (w, h))

        # Aggregation buffers
        tracks: Dict[int, TrackItem] = {}       # track_id -> TrackItem
        conf_sums: Dict[int, float] = {}
        conf_counts: Dict[int, int] = {}

        frame_idx = -1
        for r in stream:
            frame_idx += 1
            frame = r.orig_img if hasattr(r, "orig_img") else None
            if frame is None:
                # fallback: try to read from cap at same index if needed
                pass

            # results: r.boxes has .id for trackers
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                if writer is not None and frame is not None:
                    writer.write(frame)
                continue

            ids = getattr(boxes, "id", None)
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().astype(float)
            xyxys = boxes.xyxy.cpu().numpy().astype(float)

            # draw & aggregate
            for i in range(len(clss)):
                tid = int(ids[i].item()) if ids is not None else -1
                cls_id = int(clss[i])
                if tid < 0 or cls_id not in self.classes:
                    continue
                x1, y1, x2, y2 = [int(v) for v in xyxys[i]]
                conf = float(confs[i])
                tsec = frame_idx / float(fps)

                if tid not in tracks:
                    tracks[tid] = TrackItem(
                        track_id=tid,
                        cls_id=cls_id,
                        cls_name=COCO_NAMES[cls_id],
                        conf_avg=0.0,
                        start_frame=frame_idx,
                        end_frame=frame_idx,
                        start_time_sec=tsec,
                        end_time_sec=tsec,
                        points=[],
                        repr_frame_path=None,
                    )
                    conf_sums[tid] = 0.0
                    conf_counts[tid] = 0

                item = tracks[tid]
                item.end_frame = frame_idx
                item.end_time_sec = tsec
                item.points.append(TrackPoint(frame_idx, tsec, (x1, y1, x2, y2)))
                conf_sums[tid] += conf
                conf_counts[tid] += 1

                # draw
                if writer is not None and frame is not None:
                    color = _color_for_id(tid)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{item.cls_name} #{tid}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 2, cv2.LINE_AA)

            if writer is not None and frame is not None:
                writer.write(frame)

        if writer is not None:
            writer.release()

        # finalize averages + representative frames, filter short tracks
        for tid, item in list(tracks.items()):
            n = max(1, conf_counts.get(tid, 1))
            item.conf_avg = conf_sums.get(tid, 0.0) / n
            length = item.end_frame - item.start_frame + 1
            if length < self.min_track_len:
                del tracks[tid]
                continue
            mid = _midpoint_idx(item.start_frame, item.end_frame)
            rep_path = FRAMES_DIR / f"{Path(video_path).stem}.track{tid}.mid{mid}.jpg"
            item.repr_frame_path = _save_frame(video_path, mid, rep_path)

        # write tracks json
        tracks_json_path = str(OUT_DIR / (Path(video_path).stem + ".tracks.json"))
        payload = {
            "source_path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "classes": sorted(list(PERSON_AND_VEHICLES)),
            "class_names": [COCO_NAMES[i] for i in sorted(list(PERSON_AND_VEHICLES))],
            "min_track_len": self.min_track_len,
            "tracks": [
                {
                    **asdict(t),
                    "points": [asdict(p) for p in t.points],
                } for t in tracks.values()
            ],
        }
        with open(tracks_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # derive “Findings” for the report generator (Problem statement fields)
        findings = []
        for t in tracks.values():
            # take last bbox as the representative bbox
            bbox = t.points[-1].xyxy if t.points else None
            findings.append({
                "time_window": _sec_to_window(t.start_time_sec, t.end_time_sec),
                "track_id": t.track_id,
                "object_type": t.cls_name,
                "representative_frame_path": t.repr_frame_path,
                "bbox": bbox,
                "matched_offender_id": None,
                "matched_offender_name": None,
                "similarity_score": None,
                "verification_status": "unverified",
            })

        findings_json_path = str(OUT_DIR / (Path(video_path).stem + ".findings.json"))
        with open(findings_json_path, "w", encoding="utf-8") as f:
            json.dump({"source_path": video_path, "findings": findings}, f, ensure_ascii=False, indent=2)

        return TrackingResult(
            source_path=video_path,
            annotated_path=annotated_path,
            tracks_json_path=tracks_json_path,
            findings_json_path=findings_json_path,
            total_tracks=len(tracks),
            total_frames=total_frames,
            fps=fps,
        )


# ---------- CLI ----------
if __name__ == "__main__":
    """
    Example:
      python -m app.vision.tracker data/uploads/your_video.mp4
      python -m app.vision.tracker data/uploads/your_video.mp4 --model yolov8l.pt --minlen 5
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str, help="Path to a video file")
    ap.add_argument("--model", default="yolov8l.pt", type=str)
    ap.add_argument("--conf", default=0.35, type=float)
    ap.add_argument("--iou", default=0.50, type=float)
    ap.add_argument("--classes", default="person+vehicles", type=str, help="'person+vehicles' or 'all'")
    ap.add_argument("--minlen", default=5, type=int, help="Minimum track length in frames")
    ap.add_argument("--device", default=None, type=str, help="'cpu' or '0' for GPU")
    args = ap.parse_args()

    classes = PERSON_AND_VEHICLES if args.classes == "person+vehicles" else None

    tracker = YoloByteTrack(
        model_name=args.model,
        conf=args.conf,
        iou=args.iou,
        classes=classes,
        min_track_len=args.minlen,
    )
    res = tracker.track_video(args.video, device=args.device)
    print("Annotated MP4:", res.annotated_path)
    print("Tracks JSON:", res.tracks_json_path)
    print("Findings JSON:", res.findings_json_path)
    print("Tracks:", res.total_tracks, "Frames:", res.total_frames, "FPS:", res.fps)
