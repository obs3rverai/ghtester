# app/services/metadata.py
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ExifTags

# ---------- helpers ----------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm", ".m4v"}


def sha256_path(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def guess_mime(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS or (guess_mime(path) or "").startswith("image/")


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS or (guess_mime(path) or "").startswith("video/")


# ---------- EXIF (images) ----------

def _pil_exif_to_dict(img: Image.Image) -> Dict[str, Any]:
    """Converts PIL EXIF to readable dict (best-effort)."""
    exif_raw = getattr(img, "_getexif", lambda: None)()
    if not exif_raw:
        return {}

    # Map numeric EXIF tags to names
    tag_map = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif_raw.items()}

    # GPS sub-IFD normalisation (if present)
    gps_ifd = tag_map.get("GPSInfo")
    if isinstance(gps_ifd, dict):
        gps_named = {}
        for k, v in gps_ifd.items():
            name = ExifTags.GPSTAGS.get(k, str(k))
            gps_named[name] = v
        tag_map["GPSInfo"] = gps_named

    return tag_map


def extract_image_metadata(path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"type": "image", "exif": {}, "width": None, "height": None}
    try:
        with Image.open(path) as img:
            meta["width"], meta["height"] = img.size
            meta["mode"] = img.mode
            meta["format"] = img.format
            meta["exif"] = _pil_exif_to_dict(img)
    except Exception as e:
        meta["error"] = f"EXIF/preview error: {e}"
    return meta


# ---------- FFprobe (videos) ----------

def _run_ffprobe_json(path: Path) -> Dict[str, Any]:
    """
    Returns ffprobe JSON (requires ffprobe on PATH).
    If ffprobe is not installed, returns a dict with 'ffprobe_error'.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return json.loads(out.decode("utf-8", errors="ignore"))
    except FileNotFoundError:
        return {"ffprobe_error": "ffprobe not found on system PATH."}
    except subprocess.CalledProcessError as e:
        return {"ffprobe_error": f"ffprobe failed: {e.output.decode('utf-8', errors='ignore')}"}
    except Exception as e:
        return {"ffprobe_error": f"ffprobe error: {e}"}


def extract_video_metadata(path: Path) -> Dict[str, Any]:
    info = _run_ffprobe_json(path)
    meta: Dict[str, Any] = {"type": "video", "ffprobe": info}
    # convenience fields
    try:
        fmt = info.get("format", {})
        meta["duration_sec"] = float(fmt.get("duration")) if fmt.get("duration") else None
        meta["bit_rate"] = int(fmt.get("bit_rate")) if fmt.get("bit_rate") else None

        # choose first video stream for dimensions/fps
        vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if vstreams:
            v0 = vstreams[0]
            meta["width"] = v0.get("width")
            meta["height"] = v0.get("height")
            # avg_frame_rate like "30000/1001"
            afr = v0.get("avg_frame_rate")
            if afr and afr != "0/0":
                num, den = afr.split("/")
                meta["fps"] = (float(num) / float(den)) if float(den) != 0 else None
            else:
                meta["fps"] = None
            meta["codec_name"] = v0.get("codec_name")
    except Exception:
        # keep ffprobe raw if parsing fails
        pass
    return meta


# ---------- Unified forensic summary ----------

@dataclass
class ForensicSummary:
    path: str
    size_bytes: int
    sha256: str
    mime: str
    kind: str  # "image" | "video" | "other"
    details: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "mime": self.mime,
            "kind": self.kind,
            "details": self.details,
        }


def summarize_forensics(path: Path) -> ForensicSummary:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Not found: {path}")

    size = path.stat().st_size
    sha = sha256_path(path)
    mime = guess_mime(path)

    if is_image(path):
        details = extract_image_metadata(path)
        kind = "image"
    elif is_video(path):
        details = extract_video_metadata(path)
        kind = "video"
    else:
        details = {"type": "other"}
        kind = "other"

    return ForensicSummary(
        path=str(path),
        size_bytes=size,
        sha256=sha,
        mime=mime,
        kind=kind,
        details=details,
    )
