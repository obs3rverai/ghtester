# app/services/ela.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps


@dataclass
class ELAResult:
    original_path: str
    ela_path: str
    jpeg_quality: int
    mean_intensity: float
    std_intensity: float
    max_intensity: float
    high_area_fraction: float  # fraction of pixels over threshold
    threshold_used: int
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _to_bgr(img_pil: Image.Image) -> np.ndarray:
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _rescale_to_visual(img_gray: np.ndarray) -> np.ndarray:
    """Rescale gray ELA map to 0..255 for visualization."""
    m, M = float(img_gray.min()), float(img_gray.max())
    if M <= m + 1e-6:
        return np.zeros_like(img_gray, dtype=np.uint8)
    out = (255.0 * (img_gray - m) / (M - m))
    return np.clip(out, 0, 255).astype(np.uint8)


def run_ela_on_image(
    image_path: Path,
    out_dir: Path,
    jpeg_quality: int = 90,
    hi_thresh: int = 40,
) -> ELAResult:
    """
    Classic ELA:
      - Recompress as JPEG (quality q)
      - Compute absolute difference
      - Rescale for visualization
      - Simple stats + high-intensity fraction (potential tamper zones)
    Returns ELAResult and writes an ELA PNG next to reports/assets by default.
    """
    if not image_path.exists():
        raise FileNotFoundError(str(image_path))

    # Load with Pillow to avoid color surprises
    pil_orig = Image.open(image_path).convert("RGB")

    # Recompress to temporary JPEG in-memory
    tmp_jpeg = Path(out_dir) / (image_path.stem + f".q{jpeg_quality}.jpg")
    _ensure_parent(tmp_jpeg)
    pil_orig.save(tmp_jpeg, "JPEG", quality=int(jpeg_quality), optimize=True)

    pil_jpeg = Image.open(tmp_jpeg).convert("RGB")

    # Absolute difference
    diff = ImageChops.difference(pil_orig, pil_jpeg)
    # Slight blur to reduce speckle noise; not strictly necessary
    diff = diff.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Convert to grayscale for metrics
    diff_gray = ImageOps.grayscale(diff)
    diff_np = np.array(diff_gray, dtype=np.uint8)

    # Visualization rescale
    vis_np = _rescale_to_visual(diff_np)

    # Stats
    mean_int = float(diff_np.mean())
    std_int = float(diff_np.std())
    max_int = float(diff_np.max())

    # High-intensity mask fraction
    hi_mask = (vis_np >= int(hi_thresh)).astype(np.uint8)
    frac_hi = float(hi_mask.mean()) if hi_mask.size > 0 else 0.0

    # Save ELA visualization as PNG
    ela_png = Path(out_dir) / (image_path.stem + f".ELA.q{jpeg_quality}.png")
    _ensure_parent(ela_png)
    Image.fromarray(vis_np).save(ela_png)

    # Compose short notes
    notes = (
        "ELA highlights compression residuals. Localized bright regions can indicate edits, "
        "but ELA is NOT definitive. Use alongside metadata and contextual analysis."
    )

    return ELAResult(
        original_path=str(image_path),
        ela_path=str(ela_png),
        jpeg_quality=int(jpeg_quality),
        mean_intensity=mean_int,
        std_intensity=std_int,
        max_intensity=max_int,
        high_area_fraction=frac_hi,
        threshold_used=int(hi_thresh),
        notes=notes,
    )


def run_ela_on_video_frame(
    video_path: Path,
    frame_idx: int,
    out_dir: Path,
    jpeg_quality: int = 90,
    hi_thresh: int = 40,
) -> ELAResult:
    """
    Convenience for videos: extract a specific frame (BGR), write a temporary PNG, then run ELA.
    """
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    # Write the frame as PNG (lossless) for stable ELA input
    frame_png = Path(out_dir) / f"{video_path.stem}_frame{frame_idx}.png"
    _ensure_parent(frame_png)
    cv2.imwrite(str(frame_png), frame)

    return run_ela_on_image(frame_png, out_dir=out_dir, jpeg_quality=jpeg_quality, hi_thresh=hi_thresh)
