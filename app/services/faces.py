# app/services/faces.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---- InsightFace (RetinaFace + ArcFace via FaceAnalysis) ----
from insightface.app import FaceAnalysis


@dataclass
class FaceDet:
    bbox: Tuple[int, int, int, int]      # x, y, w, h (on ORIGINAL image coords)
    kps: List[Tuple[float, float]]       # 5 landmarks (original coords)
    det_score: float                     # detector confidence
    sharpness: float                     # Laplacian variance
    normed_embedding: List[float]        # L2-normalized arcface embedding
    frame_idx: Optional[int] = None
    ts_sec: Optional[float] = None


class FaceEngine:
    """
    Face detection + embedding with quality control for blurry CCTV frames.
    """
    def __init__(
        self,
        providers: Optional[List[str]] = None,
        det_size: int = 1024,                 # ↑ from 640 -> 1024 for better tiny-face recall
        rec_name: str = "arcface_r100_v1",
        min_face_size: int = 40,              # drop very tiny boxes (pixels on original image)
        min_sharpness: float = 60.0,          # Laplacian focus threshold (raise if still blurry)
        pre_upscale_if_below_height: int = 720,
        pre_upscale_factor: float = 1.5       # upsample small frames before detection/embedding
    ):
        self.providers = providers
        self.det_size = det_size
        self.rec_name = rec_name
        self.min_face_size = int(min_face_size)
        self.min_sharpness = float(min_sharpness)
        self.pre_upscale_if_below_height = int(pre_upscale_if_below_height)
        self.pre_upscale_factor = float(pre_upscale_factor)
        self._app: Optional[FaceAnalysis] = None

    def _get_app(self) -> FaceAnalysis:
        if self._app is None:
            app = FaceAnalysis(
                name="buffalo_l",            # SCRFD detector + ArcFace recognizer
                providers=self.providers
            )
            app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))
            self._app = app
        return self._app

    @staticmethod
    def _clip_bbox(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
        x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1)); y2 = max(0, min(y2, H - 1))
        return int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))

    @staticmethod
    def _l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(vec) + eps
        return (vec / n).astype(np.float32)

    @staticmethod
    def _sharpness_score(bgr_crop: np.ndarray) -> float:
        """
        Variance of Laplacian: higher = sharper. Typical CCTV sharp crops > ~80.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _maybe_upscale(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        If frame is small (height < threshold), upscale to help tiny faces.
        Returns: (possibly upscaled frame, scale factor applied).
        """
        H = frame_bgr.shape[0]
        if H < self.pre_upscale_if_below_height:
            sf = self.pre_upscale_factor
            up = cv2.resize(frame_bgr, None, fx=sf, fy=sf, interpolation=cv2.INTER_CUBIC)
            return up, sf
        return frame_bgr, 1.0

    def faces_from_bgr(self, frame_bgr: np.ndarray, frame_idx: Optional[int] = None, ts_sec: Optional[float] = None) -> List[FaceDet]:
        """
        Detect faces and compute embeddings from a BGR frame.
        Quality improvements:
          - optional pre-upscale for small frames
          - drop boxes < min_face_size
          - drop soft faces with Laplacian variance < min_sharpness
        All returned bboxes/landmarks are mapped to ORIGINAL frame coordinates.
        """
        H0, W0 = frame_bgr.shape[:2]
        app = self._get_app()

        # 1) Optional upscale for tiny faces
        work_bgr, sf = self._maybe_upscale(frame_bgr)
        H, W = work_bgr.shape[:2]

        # 2) Detect+embed on upscaled frame
        persons = app.get(work_bgr)

        out: List[FaceDet] = []
        for p in persons:
            x1, y1, x2, y2 = [int(v) for v in p.bbox]  # xyxy on WORK image
            # map back to ORIGINAL coords
            if sf != 1.0:
                x1 = int(round(x1 / sf)); y1 = int(round(y1 / sf))
                x2 = int(round(x2 / sf)); y2 = int(round(y2 / sf))
            x, y, w, h = self._clip_bbox(x1, y1, x2, y2, W0, H0)

            # 3) size filter (on original coords)
            if min(w, h) < self.min_face_size:
                continue

            # 4) sharpness on ORIGINAL crop
            crop = frame_bgr[max(0, y):min(y+h, H0), max(0, x):min(x+w, W0), :]
            sharp = self._sharpness_score(crop)
            if sharp < self.min_sharpness:
                continue

            # 5) embedding (InsightFace already provides normed embedding)
            if hasattr(p, "normed_embedding") and p.normed_embedding is not None:
                emb = np.array(p.normed_embedding, dtype=np.float32)
            else:
                emb = np.array(p.embedding, dtype=np.float32)
                emb = self._l2_normalize(emb)

            # map landmarks back to original coords too
            kps = []
            for (xx, yy) in p.kps:
                if sf != 1.0:
                    xx = float(xx) / sf
                    yy = float(yy) / sf
                kps.append((float(xx), float(yy)))

            out.append(FaceDet(
                bbox=(x, y, w, h),
                kps=kps,
                det_score=float(p.det_score),
                sharpness=sharp,
                normed_embedding=emb.tolist(),
                frame_idx=frame_idx,
                ts_sec=ts_sec
            ))
        return out

    # ---------- High-level helpers ----------

    def extract_from_image(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(str(path))
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        faces = self.faces_from_bgr(img)
        H, W = img.shape[:2]
        return {
            "image_path": str(path),
            "width": W,
            "height": H,
            "faces": [asdict(f) for f in faces],
        }

    def extract_from_video(self, path: Path, sample_fps: float = 2.0, max_frames: Optional[int] = 300) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(str(path))

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        stride = max(1, int(round(fps_in / max(0.1, sample_fps))))

        frame_idx = -1
        processed = 0
        faces_all: List[Dict[str, Any]] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % stride != 0:
                continue

            ts_sec = frame_idx / float(max(1e-6, fps_in))
            fdet = self.faces_from_bgr(frame, frame_idx=frame_idx, ts_sec=ts_sec)
            faces_all.extend([asdict(f) for f in fdet])
            processed += 1
            if max_frames is not None and processed >= max_frames:
                break

        cap.release()
        return {
            "video_path": str(path),
            "fps_in": float(fps_in),
            "fps_sampling": float(sample_fps),
            "processed_frames": processed,
            "width": W,
            "height": H,
            "faces": faces_all
        }


# ---------- Convenience singleton ----------
_default_engine: Optional[FaceEngine] = None

def get_default_face_engine() -> FaceEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = FaceEngine(
            providers=None,        # let ORT pick GPU/CPU
            det_size=1024,         # bigger detector input
            min_face_size=40,      # drop tiny boxes
            min_sharpness=60.0,    # drop soft faces
            pre_upscale_if_below_height=720,
            pre_upscale_factor=1.5
        )
    return _default_engine


# ---------- Top-level helpers ----------
def faces_from_image(path: Path) -> Dict[str, Any]:
    return get_default_face_engine().extract_from_image(path)

def faces_from_video(path: Path, sample_fps: float = 2.0, max_frames: Optional[int] = 300) -> Dict[str, Any]:
    return get_default_face_engine().extract_from_video(path, sample_fps=sample_fps, max_frames=max_frames)
