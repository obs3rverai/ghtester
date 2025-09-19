# app/vision/detector.py
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ---------- paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
IN_DIR       = DATA_DIR / "uploads"
OUT_DIR      = DATA_DIR / "detections"
FRAMES_DIR   = DATA_DIR / "frames"
ANNOT_DIR    = DATA_DIR / "annotated"

for p in [OUT_DIR, FRAMES_DIR, ANNOT_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ---------- classes / labels ----------
# COCO indices (Ultralytics YOLO)
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
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

PERSON_AND_VEHICLES = {0, 1, 2, 3, 5, 7}  # person + bikes + car + motorcycle + bus + truck


# ---------- dataclasses ----------
@dataclass
class Detection:
    frame_index: int
    cls: str
    conf: float
    xyxy: Tuple[int, int, int, int]  # x1,y1,x2,y2


@dataclass
class DetectionResult:
    source_path: str
    annotated_path: Optional[str]
    json_path: str
    num_frames: int
    total_detections: int
    per_frame: List[Detection]


# ---------- drawing helpers ----------
def _color_for_class(cls_id: int) -> Tuple[int, int, int]:
    # deterministic “nice” colors per class
    rng = np.random.default_rng(seed=cls_id * 7919)
    return tuple(int(x) for x in rng.integers(50, 230, 3))  # BGR


def _draw_box(
    frame: np.ndarray,
    xyxy: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    # label box
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 15, 15), 1, cv2.LINE_AA)


# ---------- core detector ----------
class YoloDetector:
    """
    YOLO detector for images/videos.
    - Defaults to yolov8l for higher accuracy (good on CPU, still not tiny).
    - You can switch to 'yolov10x.pt' if you enable GPU and want SOTA accuracy.
    """
    def __init__(
        self,
        model_name: str = "yolov8l.pt",
        conf: float = 0.35,
        iou: float = 0.50,
        classes: Optional[set[int]] = None,
    ):
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self.classes = PERSON_AND_VEHICLES if classes is None else classes

    # ----------- image -----------
    def detect_image(
        self,
        image_path: str | Path,
        annotate: bool = True,
        save_name: Optional[str] = None,
    ) -> DetectionResult:
        image_path = str(image_path)
        results = self.model.predict(
            image_path,
            conf=self.conf,
            iou=self.iou,
            classes=sorted(self.classes),
            verbose=False
        )
        dets: List[Detection] = []
        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError(f"Could not load image: {image_path}")

        for r in results:
            # single frame result
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for b in r.boxes:
                cls_id = int(b.cls.item())
                conf = float(b.conf.item())
                x1, y1, x2, y2 = map(int, b.xyxy.cpu().numpy()[0])
                label = f"{COCO_NAMES[cls_id]} {conf:.2f}"
                dets.append(Detection(0, COCO_NAMES[cls_id], conf, (x1, y1, x2, y2)))
                if annotate:
                    _draw_box(frame, (x1, y1, x2, y2), label, _color_for_class(cls_id))

        # save annotated
        annotated_path = None
        if annotate:
            out_name = save_name or (Path(image_path).stem + ".det.jpg")
            annotated_path = str(ANNOT_DIR / out_name)
            cv2.imwrite(annotated_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        # save json
        json_path = str(OUT_DIR / (Path(image_path).stem + ".detections.json"))
        _save_dets_json(json_path, image_path, dets, num_frames=1)

        return DetectionResult(
            source_path=image_path,
            annotated_path=annotated_path,
            json_path=json_path,
            num_frames=1,
            total_detections=len(dets),
            per_frame=dets,
        )

    # ----------- video -----------
    def detect_video(
        self,
        video_path: str | Path,
        annotate: bool = True,
        max_frames: Optional[int] = None,
        stride: int = 1,
        out_name: Optional[str] = None,
    ) -> DetectionResult:
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        annotated_path = None
        if annotate:
            out_name = out_name or (Path(video_path).stem + ".det.mp4")
            annotated_path = str(ANNOT_DIR / out_name)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(annotated_path, fourcc, fps, (w, h))

        dets_all: List[Detection] = []
        frame_idx = -1
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % stride != 0:
                continue
            processed += 1
            if max_frames is not None and processed > max_frames:
                break

            # YOLO inference on this frame (numpy array)
            res = self.model.predict(
                frame,
                conf=self.conf,
                iou=self.iou,
                classes=sorted(self.classes),
                verbose=False
            )[0]

            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls_id = int(b.cls.item())
                    conf = float(b.conf.item())
                    x1, y1, x2, y2 = map(int, b.xyxy.cpu().numpy()[0])
                    label = f"{COCO_NAMES[cls_id]} {conf:.2f}"
                    dets_all.append(Detection(frame_idx, COCO_NAMES[cls_id], conf, (x1, y1, x2, y2)))
                    if annotate and writer is not None:
                        _draw_box(frame, (x1, y1, x2, y2), label, _color_for_class(cls_id))

            if annotate and writer is not None:
                writer.write(frame)

        cap.release()
        if writer is not None:
            writer.release()

        # save json
        json_path = str(OUT_DIR / (Path(video_path).stem + ".detections.json"))
        _save_dets_json(json_path, video_path, dets_all, num_frames=frame_idx + 1)

        return DetectionResult(
            source_path=video_path,
            annotated_path=annotated_path,
            json_path=json_path,
            num_frames=frame_idx + 1,
            total_detections=len(dets_all),
            per_frame=dets_all,
        )


def _save_dets_json(json_path: str, src_path: str, dets: List[Detection], num_frames: int) -> None:
    import json
    payload = {
        "source_path": src_path,
        "num_frames": num_frames,
        "classes": sorted(list(PERSON_AND_VEHICLES)),
        "class_names": [COCO_NAMES[i] for i in sorted(list(PERSON_AND_VEHICLES))],
        "detections": [asdict(d) for d in dets],
        "schema": {
            "detection": ["frame_index", "cls", "conf", "xyxy"]
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------- quick CLI tests ----------
if __name__ == "__main__":
    """
    Example:
      python -m app.vision.detector /path/to/image.jpg
      python -m app.vision.detector /path/to/video.mp4 --video --max 200 --stride 2
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=str, help="Image or video path")
    ap.add_argument("--model", default="yolov8l.pt", type=str)
    ap.add_argument("--conf", default=0.35, type=float)
    ap.add_argument("--iou", default=0.50, type=float)
    ap.add_argument("--classes", default="person+vehicles", type=str, help="'person+vehicles' or 'all'")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--max", type=int, default=None, help="Max frames to process (video)")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame (video)")
    args = ap.parse_args()

    classes = PERSON_AND_VEHICLES if args.classes == "person+vehicles" else None

    det = YoloDetector(model_name=args.model, conf=args.conf, iou=args.iou, classes=classes)

    if args.video:
        res = det.detect_video(args.path, annotate=True, max_frames=args.max, stride=args.stride)
    else:
        res = det.detect_image(args.path, annotate=True)

    print("Annotated:", res.annotated_path)
    print("JSON:", res.json_path)
    print("Total detections:", res.total_detections)
