# ui/dashboard.py
from __future__ import annotations

import os, json, mimetypes, datetime, uuid
from pathlib import Path
from typing import Dict, Any, List, DefaultDict, Tuple
from collections import defaultdict

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

# =========================
# Basic config
# =========================
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")  # with or without /api suffix
PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="AI CCTV & Digital Media Forensic Tool — MVP", layout="wide")
st.title("AI CCTV & Digital Media Forensic Tool — MVP")
st.caption("Hackathon demo UI (uploads, forensics, motion/YOLO detection, faces index & search).")

# =========================
# HTTP helpers
# =========================
def _api_base_api() -> str:
    """Return a base that ends with /api for backends mounted under /api."""
    return API_BASE if API_BASE.rstrip("/").endswith("/api") else API_BASE.rstrip("/") + "/api"

def api_health() -> dict:
    r = requests.get(f"{API_BASE}/health", timeout=5); r.raise_for_status(); return r.json()

def api_ingest(file_bytes: bytes, filename: str) -> dict:
    files = {"file": (filename, file_bytes, mimetypes.guess_type(filename)[0] or "application/octet-stream")}
    r = requests.post(f"{API_BASE}/ingest", files=files, timeout=60); r.raise_for_status(); return r.json()

def api_list_files() -> dict:
    url1 = f"{API_BASE}/files"; url2 = f"{_api_base_api()}/files"
    r = requests.get(url1, timeout=10)
    if r.status_code == 404: r = requests.get(url2, timeout=10)
    r.raise_for_status(); return r.json()

def api_delete_file(stored_name: str, deep: bool = False) -> dict:
    params = {"deep": str(bool(deep)).lower()}
    url1 = f"{API_BASE}/files/{stored_name}"; url2 = f"{_api_base_api()}/files/{stored_name}"
    r = requests.delete(url1, params=params, timeout=30)
    if r.status_code == 404: r = requests.delete(url2, params=params, timeout=30)
    r.raise_for_status(); return r.json()

def api_forensics_files() -> dict:
    r = requests.get(f"{API_BASE}/forensics/files", timeout=10); r.raise_for_status(); return r.json()

def api_forensics_by_filename(stored_name: str) -> dict:
    r = requests.get(f"{API_BASE}/forensics/summary/by-filename/{stored_name}", timeout=60); r.raise_for_status(); return r.json()

def api_detect_options() -> dict:
    r = requests.get(f"{API_BASE}/detect/options", timeout=10); r.raise_for_status(); return r.json()

def api_detect_by_filename(stored_name: str, params: dict) -> dict:
    r = requests.post(f"{API_BASE}/detect/run/by-filename/{stored_name}", json=params, timeout=1200)
    r.raise_for_status(); return r.json()

# Faces APIs
def api_faces_extract_by_filename(stored_name: str, sample_fps: float, max_frames: int) -> dict:
    r = requests.post(f"{API_BASE}/faces/extract/by-filename/{stored_name}",
                      json={"sample_fps": float(sample_fps), "max_frames": int(max_frames)}, timeout=600)
    r.raise_for_status(); return r.json()

def api_faces_index_reset() -> dict:
    r = requests.post(f"{API_BASE}/faces/index/reset", timeout=10); r.raise_for_status(); return r.json()

def api_faces_index_add_by_filename(stored_name: str, sample_fps: float, max_frames: int, max_faces: int) -> dict:
    r = requests.post(f"{API_BASE}/faces/index/add/by-filename/{stored_name}",
                      json={"sample_fps": float(sample_fps), "max_frames": int(max_frames), "max_faces": int(max_faces)},
                      timeout=1200)
    r.raise_for_status(); return r.json()

def api_faces_index_stats() -> dict:
    r = requests.get(f"{API_BASE}/faces/index/stats", timeout=10); r.raise_for_status(); return r.json()

def api_faces_search(file_bytes: bytes, filename: str, top_k: int) -> dict:
    files = {"file": (filename, file_bytes, mimetypes.guess_type(filename)[0] or "application/octet-stream")}
    data = {"top_k": str(int(top_k))}
    r = requests.post(f"{API_BASE}/faces/search", files=files, data=data, timeout=120)
    r.raise_for_status(); return r.json()

def _sanitize_embedding(vec) -> List[float]:
    out: List[float] = []
    for v in (vec or []):
        try: f = float(v)
        except Exception: continue
        if np.isfinite(f): out.append(f)
    return out

def api_faces_search_by_embedding(embedding: List[float], top_k: int) -> dict:
    payload = {"embedding": _sanitize_embedding(embedding), "top_k": int(top_k)}
    if not payload["embedding"]: raise ValueError("Empty/invalid embedding vector.")
    url1 = f"{API_BASE}/faces/search/by-embedding"; url2 = f"{_api_base_api()}/faces/search/by-embedding"
    r = requests.post(url1, json=payload, timeout=60)
    if r.status_code == 404: r = requests.post(url2, json=payload, timeout=60)
    r.raise_for_status(); return r.json()

def api_ela_by_filename(stored_name: str, jpeg_quality: int, hi_thresh: int, frame_idx: int | None) -> dict:
    params = {"jpeg_quality": int(jpeg_quality), "hi_thresh": int(hi_thresh)}
    if frame_idx is not None: params["frame_idx"] = int(frame_idx)
    r = requests.post(f"{API_BASE}/forensics/ela/by-filename/{stored_name}", params=params, timeout=120)
    r.raise_for_status(); return r.json()

def _load_json_file(path_like: str | None) -> dict:
    if not path_like:
        return {}
    p = Path(path_like)
    if not p.is_absolute():
        p = (PROJECT_ROOT / path_like).resolve()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def api_report_generate(payload: dict) -> dict:
    r = requests.post(f"{API_BASE}/report/generate", json=payload, timeout=120)
    r.raise_for_status(); return r.json()

# Verify endpoints (UI tab will call /api/verify/pdf)
def api_verify_pdf(pdf_bytes: bytes, public_key_pem: bytes | None, bundle_json: bytes | None) -> dict:
    api_base_api = _api_base_api()
    files = {"pdf": ("report.pdf", pdf_bytes, "application/pdf")}
    if public_key_pem: files["public_key_pem"] = ("public.pem", public_key_pem, "application/x-pem-file")
    if bundle_json: files["bundle_json"] = ("bundle.json", bundle_json, "application/json")
    r = requests.post(f"{api_base_api}/verify/pdf", files=files, timeout=120)
    r.raise_for_status(); return r.json()

# =========================
# Local preview & IO helpers
# =========================
def try_show_media(stored_path: str):
    abs_path = (PROJECT_ROOT / stored_path).resolve()
    if not abs_path.exists():
        st.info("File saved, but preview not found on this path."); return
    suffix = abs_path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        st.image(Image.open(abs_path), caption=abs_path.name, use_container_width=True)
    elif suffix in {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}:
        try: st.video(str(abs_path))
        except Exception:
            with abs_path.open("rb") as f: st.video(f.read())
    else:
        st.write("Preview not supported for this file type.")

def draw_overlays_on_frame(video_path: str, frame_idx: int, dets_this_frame: List[Dict[str, Any]]) -> Image.Image:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise RuntimeError(f"Could not open video for preview: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx); ok, frame = cap.read(); cap.release()
    if not ok or frame is None: raise RuntimeError(f"Could not read frame {frame_idx}")
    for d in dets_this_frame:
        x, y, w, h = d["bbox"]; x2, y2 = x+w, y+h
        cv2.rectangle(frame, (x,y), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{d.get('label','obj')} {d.get('conf',0):.2f}", (x, max(0,y-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def group_detections_by_frame(detections: List[Dict[str, Any]]) -> DefaultDict[int, List[Dict[str, Any]]]:
    grouped: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for d in detections: grouped[int(d["frame_idx"])].append(d)
    return grouped

def crop_from_source(source_file: str, bbox: List[int], frame_idx: int | None) -> Image.Image | None:
    src_path = (PROJECT_ROOT / "data" / "uploads" / source_file).resolve()
    if not src_path.exists(): return None
    x, y, w, h = map(int, bbox)
    if src_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        img = cv2.imread(str(src_path)); 
        if img is None: return None
        H, W = img.shape[:2]; x2, y2 = min(x+w, W-1), min(y+h, H-1)
        crop = img[max(0,y):y2, max(0,x):x2, :]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    else:
        if frame_idx is None: return None
        cap = cv2.VideoCapture(str(src_path)); 
        if not cap.isOpened(): return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx)); ok, frame = cap.read(); cap.release()
        if not ok or frame is None: return None
        H, W = frame.shape[:2]; x2, y2 = min(x+w, W-1), min(y+h, H-1)
        crop = frame[max(0,y):y2, max(0,x):x2, :]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _save_snapshot(source_file: str, bbox: List[int] | None, frame_idx: int | None, kind: str, label: str | None) -> str | None:
    """Create a thumbnail crop for report; returns path relative to project root (str)."""
    thumbs_dir = PROJECT_ROOT / "data" / "findings" / "thumbs"
    _ensure_dir(thumbs_dir)
    stem = Path(source_file).stem
    uid = uuid.uuid4().hex[:8]
    name = f"{stem}_{kind}_{(label or 'obj')}_{frame_idx if frame_idx is not None else 0}_{uid}.jpg"
    out_path = thumbs_dir / name
    if bbox is None:
        # try to grab a central crop if possible (fallback)
        img = crop_from_source(source_file, [0, 0, 256, 256], frame_idx)
    else:
        img = crop_from_source(source_file, bbox, frame_idx)
    if img is None:
        return None
    try:
        img.save(out_path, format="JPEG", quality=90)
        return str(out_path.relative_to(PROJECT_ROOT))
    except Exception:
        return None

# =========================
# Findings synthesizer (client-side fallback)
# =========================
def _label_to_object_type(lbl: str) -> str:
    lbl = (lbl or "").lower()
    if any(k in lbl for k in ["person", "pedestrian", "face", "human"]): return "person"
    if any(k in lbl for k in ["car", "vehicle", "truck", "bus", "bike", "motor", "auto", "van", "taxi"]): return "car"
    return "object"

def _fmt_time_window(start_f: int, end_f: int, fps: float | int | None) -> str:
    if not fps or fps <= 0: return f"frames {start_f}-{end_f}"
    s0, s1 = start_f / float(fps), end_f / float(fps)
    return f"{s0:.2f}-{s1:.2f}s"

def synthesize_findings_from_result(res: Dict[str, Any], source_filename: str, max_findings: int = 20) -> List[Dict[str, Any]]:
    meta = res.get("meta", {}) or {}
    dets: List[Dict[str, Any]] = res.get("detections", []) or []
    tracks: List[Dict[str, Any]] = res.get("tracks", []) or []
    fps = meta.get("fps") or meta.get("fps_sampling") or meta.get("video_fps") or None
    findings: List[Dict[str, Any]] = []

    # Strategy A: per-track summary
    if tracks:
        by_tid: Dict[Any, List[Dict[str, Any]]] = {}
        for d in dets:
            tid = d.get("track_id")
            if tid is None: continue
            by_tid.setdefault(tid, []).append(d)

        for t in tracks:
            tid = t.get("id", t.get("track_id"))
            if tid is None: continue
            t_dets = by_tid.get(tid, [])
            if t_dets:
                frames = [int(x.get("frame_idx", 0)) for x in t_dets]
                start_f, end_f = min(frames), max(frames)
                mid_f = (start_f + end_f) // 2
                mid_det = min(t_dets, key=lambda x: abs(int(x.get("frame_idx", 0)) - mid_f))
                bbox = mid_det.get("bbox")
                label = t.get("label") or mid_det.get("label") or "object"
            else:
                start_f = int(t.get("start_frame", 0)); end_f = int(t.get("end_frame", start_f))
                bbox = None; label = t.get("label") or "object"

            obj_type = _label_to_object_type(label)
            time_window = _fmt_time_window(start_f, end_f, fps)
            findings.append({
                "time_window": time_window,
                "track_id": int(tid) if isinstance(tid, (int, float)) else None,
                "object_type": obj_type,
                "representative_frame_path": "",
                "bbox": bbox,
                "matched_offender_id": None,
                "matched_offender_name": None,
                "similarity_score": None,
                "verification_status": "unverified",
                "source_file": source_filename,
            })
            if len(findings) >= max_findings: break

    # Strategy B: top detections when no tracks
    if not findings and dets:
        dets_sorted = sorted(dets, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
        seen_frames: set[int] = set()
        for d in dets_sorted:
            fidx = int(d.get("frame_idx", 0))
            if fidx in seen_frames: continue
            seen_frames.add(fidx)
            label = d.get("label") or "object"
            obj_type = _label_to_object_type(label)
            time_window = _fmt_time_window(fidx, fidx, fps)
            findings.append({
                "time_window": time_window,
                "track_id": d.get("track_id") if isinstance(d.get("track_id"), int) else None,
                "object_type": obj_type,
                "representative_frame_path": "",
                "bbox": d.get("bbox"),
                "matched_offender_id": None,
                "matched_offender_name": None,
                "similarity_score": None,
                "verification_status": "unverified",
                "source_file": source_filename,
            })
            if len(findings) >= max_findings: break

    return findings

def add_synth_findings_to_report_session(res: Dict[str, Any], source_filename: str) -> int:
    synth = synthesize_findings_from_result(res, source_filename)
    if not synth: return 0
    st.session_state.setdefault("rep_findings", [])
    before = len(st.session_state["rep_findings"])
    st.session_state["rep_findings"].extend(synth)
    return len(st.session_state["rep_findings"]) - before

# =========================
# NEW: Analysis Candidates (for Report)
# =========================
def _calc_timestamp_sec(frame_idx: int | None, fps: float | int | None) -> float | None:
    if frame_idx is None or not fps or fps <= 0: return None
    return float(frame_idx) / float(fps)

def _make_candidate(source_file: str, kind: str, label: str | None, conf: float | None,
                    frame_idx: int | None, fps: float | int | None, bbox: List[int] | None,
                    snapshot: str | None, include: bool = True) -> Dict[str, Any]:
    return {
        "id": uuid.uuid4().hex[:12],
        "type": kind,  # "person" | "vehicle" | "face" | "object"
        "source_file": source_file,
        "label": label,
        "confidence": None if conf is None else float(conf),
        "frame_idx": None if frame_idx is None else int(frame_idx),
        "timestamp_sec": _calc_timestamp_sec(frame_idx, fps),
        "bbox": bbox,
        "snapshot_path": snapshot,  # relative to project root
        "include": bool(include),
    }

def _label_to_kind_for_candidate(lbl: str | None) -> str:
    lbl = (lbl or "").lower()
    if "person" in lbl or "pedestrian" in lbl or "face" in lbl or "human" in lbl: return "person"
    if any(k in lbl for k in ["car", "vehicle", "truck", "bus", "bike", "motor", "auto", "van", "taxi"]): return "vehicle"
    return "object"

def add_candidates_from_detection_result(
    res: Dict[str, Any],
    source_filename: str,
    make_thumbs: bool = True,
    max_items: int = 60,
) -> int:
    """Harvest detections/tracks into candidates; return count added."""
    meta = res.get("meta", {}) or {}
    dets: List[Dict[str, Any]] = res.get("detections", []) or []
    tracks: List[Dict[str, Any]] = res.get("tracks", []) or []
    fps = meta.get("fps") or meta.get("fps_sampling") or meta.get("video_fps") or None

    st.session_state.setdefault("rep_candidates", [])
    added = 0

    # Prefer tracks (each track -> one candidate at midpoint)
    if tracks:
        det_by_track: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for d in dets:
            tid = d.get("track_id")
            if tid is not None:
                det_by_track[tid].append(d)

        for t in tracks:
            tid = t.get("id", t.get("track_id"))
            if tid is None:
                continue
            t_dets = det_by_track.get(tid, [])
            if t_dets:
                frames = [int(x.get("frame_idx", 0)) for x in t_dets]
                start_f, end_f = min(frames), max(frames)
                mid_f = (start_f + end_f) // 2
                mid_det = min(t_dets, key=lambda x: abs(int(x.get("frame_idx", 0)) - mid_f))
                bbox = mid_det.get("bbox")
                label = t.get("label") or mid_det.get("label") or "object"
                conf = mid_det.get("conf")
            else:
                bbox = None
                label = t.get("label") or "object"
                conf = None
                mid_f = t.get("mid_frame") or t.get("start_frame")

            kind = _label_to_kind_for_candidate(label)
            snap = _save_snapshot(
                source_filename, bbox,
                int(mid_f) if mid_f is not None else None,
                kind, label
            ) if make_thumbs else None

            st.session_state["rep_candidates"].append(
                _make_candidate(
                    source_filename, kind, label, conf,
                    int(mid_f) if mid_f is not None else None,
                    fps, bbox, snap, True
                )
            )
            added += 1
            if added >= max_items:
                return added

    # If no tracks, take top-N detections across diverse frames
    if added == 0 and dets:
        dets_sorted = sorted(dets, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
        seen_frames: set[int] = set()
        for d in dets_sorted:
            fidx = int(d.get("frame_idx", 0))
            if fidx in seen_frames:
                continue
            seen_frames.add(fidx)
            label = d.get("label") or "object"
            conf = d.get("conf")
            bbox = d.get("bbox")
            kind = _label_to_kind_for_candidate(label)

            snap = _save_snapshot(
                source_filename, bbox, fidx, kind, label
            ) if make_thumbs else None

            st.session_state["rep_candidates"].append(
                _make_candidate(
                    source_filename, kind, label, conf,
                    fidx, fps, bbox, snap, True
                )
            )
            added += 1
            if added >= max_items:
                break

    return added

def add_candidates_from_face_matches(matches: List[Dict[str, Any]], default_source: str | None = None, make_thumbs: bool = True, max_items: int = 40) -> int:
    st.session_state.setdefault("rep_candidates", [])
    added = 0
    for it in matches:
        src = it.get("source_file") or it.get("file") or default_source
        if not src: continue
        bbox = it.get("bbox"); frame_idx = it.get("frame_idx"); conf = it.get("score") or it.get("similarity")
        label = it.get("name") or it.get("label") or "face"
        snap = _save_snapshot(src, bbox, frame_idx, "face", label) if (make_thumbs and bbox is not None) else None
        st.session_state["rep_candidates"].append(_make_candidate(src, "face", label, conf, frame_idx, None, bbox, snap, True))
        added += 1
        if added >= max_items: break
    return added

def _candidates_to_report_findings(selected_only: bool = True) -> List[Dict[str, Any]]:
    """Map candidates to the backend report finding schema."""
    out: List[Dict[str, Any]] = []
    cands = st.session_state.get("rep_candidates", []) or []
    for c in cands:
        if selected_only and not c.get("include", True): continue
        # map type -> object_type; treat "face" as "person" for report
        obj_type = "person" if c.get("type") == "face" else (c.get("type") or "object")
        # time_window: frame idx alone, seconds if available
        tw = ""
        if c.get("timestamp_sec") is not None: tw = f"{c['timestamp_sec']:.2f}s"
        elif c.get("frame_idx") is not None: tw = f"frame {c['frame_idx']}"
        out.append({
            "time_window": tw,
            "track_id": None,
            "object_type": obj_type,
            "representative_frame_path": c.get("snapshot_path") or "",
            "bbox": c.get("bbox"),
            "matched_offender_id": None,
            "matched_offender_name": c.get("label") if obj_type == "person" else None,
            "similarity_score": c.get("confidence"),
            "verification_status": "unverified",
            "source_file": c.get("source_file"),
        })
    return out

# =========================
# Header
# =========================
health_col, files_col = st.columns([1, 2])
with health_col:
    st.subheader("Server Health")
    try:
        health = api_health()
        yolo_avail = health.get("yolo_available", "no")
        st.success(f"Backend is running ✅  |  YOLO: {yolo_avail}" if health.get("status") == "ok" else f"Backend issue: {health.get('status')}")
    except Exception as e:
        st.error(f"Cannot reach backend: {e}")

with files_col:
    st.subheader("Stored Files")
    if st.button("Refresh list", key="refresh_files"): pass
    listing = api_list_files(); files = listing.get("files", [])
    st.write(f"Count: **{listing.get('count', 0)}**")
    if not files:
        st.info("No files yet. Upload one below.")
    else:
        for f in files:
            row = st.container(border=True)
            c1, c2, c3, c4 = row.columns([6, 2, 2, 3])
            ext = Path(f).suffix.lower()
            icon = "🎥" if ext in {".mp4",".mov",".mkv",".avi",".m4v",".webm"} else "🖼️"
            c1.markdown(f"{icon} `{f}`")
            preview = c2.button("Open", key=f"open_{f}")
            deepdel = c3.checkbox("deep", value=False, key=f"deep_{f}", help="Also delete annotated/results/tracks/findings")
            delete = c4.button("Delete", type="secondary", key=f"del_{f}")
            if preview:
                try:
                    summ = api_forensics_by_filename(f)
                    rel_path = summ.get("path")
                    if rel_path:
                        rp = Path(rel_path); rp = rp if rp.is_absolute() else (PROJECT_ROOT / rel_path).resolve()
                        st.caption(f"Path: `{rp}`"); try_show_media(str(rp.relative_to(PROJECT_ROOT)))
                except Exception as e:
                    st.warning(f"Could not preview: {e}")
            if delete:
                with row:
                    st.warning(f"Type the exact filename to confirm delete of `{f}`")
                    confirm = st.text_input("Confirm name", key=f"confirm_{f}")
                    go = st.button("Confirm delete", key=f"go_{f}", type="primary")
                    if go and confirm == f:
                        try:
                            res = api_delete_file(f, deepdel); st.success(f"Deleted {res.get('deleted')}. Related: {len(res.get('related_deleted', []))}")
                            st.rerun()
                        except requests.HTTPError as e:
                            status = getattr(e.response, "status_code", "HTTP")
                            detail = ""
                            try: detail = e.response.text
                            except Exception: pass
                            st.error(f"Delete failed ({status}): {detail or 'See server logs.'}")

st.markdown("---")

# =========================
# Tabs (Verify is right-most)
# =========================
tab_upload, tab_forensics, tab_detect, tab_faces, tab_report, tab_verify = st.tabs(
    ["📤 Upload & Hash", "🧪 Forensics Metadata", "🎯 Detection", "👤 Faces (Index & Search)", "📄 Report & ELA", "🔒 Verify PDF"]
)

# --------- Upload Tab ---------
with tab_upload:
    st.subheader("Upload & Hash")
    uploaded = st.file_uploader("Select an image/video to ingest",
                                type=["jpg","jpeg","png","bmp","webp","mp4","mov","mkv","avi","m4v","webm"],
                                key="uploader_main")
    if uploaded is not None:
        col_a, _ = st.columns([1, 1])
        with col_a:
            st.write("**Selected file:**", uploaded.name)
            if uploaded.type.startswith("image/"): st.image(uploaded, caption="Preview", use_container_width=True)
            elif uploaded.type.startswith("video/"): st.info("Video selected (preview after upload).")
        if st.button("Upload & Compute SHA-256", type="primary", key="upload_and_hash_btn"):
            resp = api_ingest(uploaded.getvalue(), uploaded.name)
            st.success("Uploaded successfully!"); st.json(resp)
            st.markdown("**Local Preview (from stored path):**"); try_show_media(resp.get("stored_path", ""))

    st.info(
        "Large file? If the browser upload fails with 413, either drag the file into "
        "`data/uploads/` in the VS Code Explorer and click **Stored Files → Refresh**, or upload from the terminal inside the Codespace:\n\n"
        "```bash\n"
        "curl -F \"file=@/workspaces/ghtest/data/uploads/yourvideo.mp4\" http://127.0.0.1:8000/ingest\n"
        "```"
    )

# --------- Forensics Tab ---------
with tab_forensics:
    st.subheader("Forensic Metadata Viewer")
    left, right = st.columns([2, 3])
    with left:
        f_listing = api_forensics_files(); filenames: List[str] = f_listing.get("files", [])
        selected = st.selectbox("Stored filename", options=["-- select --"] + filenames, index=0, key="forensics_file_sel")
        cols = st.columns([1,1])
        load = cols[0].button("Load metadata", type="primary", disabled=(selected=="-- select --"), key="forensics_load_btn")
        do_del = cols[1].button("Delete file", disabled=(selected=="-- select --"), key="forensics_del_btn")
        deep = st.checkbox("Deep delete related artifacts", value=False, key="forensics_del_deep")
        if load: st.session_state["_selected_forensics_file"] = selected
        if do_del and selected != "-- select --":
            try: res = api_delete_file(selected, deep); st.success(f"Deleted {res.get('deleted')}"); st.rerun()
            except requests.HTTPError as e:
                status = getattr(e.response, "status_code", "HTTP"); detail = ""
                try: detail = e.response.text
                except Exception: pass
                st.error(f"Delete failed ({status}): {detail or 'See server logs.'}")

    with right:
        sel = st.session_state.get("_selected_forensics_file")
        if sel and sel != "-- select --":
            summary = api_forensics_by_filename(sel)
            st.markdown(f"**File:** `{sel}`")
            base_info_cols = st.columns(4)
            base_info_cols[0].metric("Type", summary.get("kind"))
            base_info_cols[1].metric("MIME", summary.get("mime"))
            size_b = summary.get("size_bytes", 0); base_info_cols[2].metric("Size (bytes)", f"{size_b:,}")
            base_info_cols[3].markdown(f"`SHA-256`:\n\n`{summary.get('sha256')}`")
            st.divider()
            rel_path = summary.get("path")
            if rel_path:
                try_show_media(Path(rel_path).relative_to(PROJECT_ROOT) if rel_path.startswith(str(PROJECT_ROOT)) else rel_path)
            details = summary.get("details", {}); kind = summary.get("kind")
            if kind == "image":
                st.markdown("### Image Details")
                img_cols = st.columns(3)
                img_cols[0].write(f"**Width:** {details.get('width')}")
                img_cols[1].write(f"**Height:** {details.get('height')}")
                img_cols[2].write(f"**Format/Mode:** {details.get('format')}/{details.get('mode')}")
                exif = details.get("exif", {}); 
                if exif: st.markdown("#### EXIF"); st.json(exif)
                if "error" in details: st.warning(details["error"])
            elif kind == "video":
                st.markdown("### Video Details")
                v_cols = st.columns(4)
                v_cols[0].write(f"**Duration (s):** {details.get('duration_sec')}")
                v_cols[1].write(f"**Bitrate:** {details.get('bit_rate')}")
                v_cols[2].write(f"**Resolution:** {details.get('width')}×{details.get('height')}")
                v_cols[3].write(f"**FPS:** {details.get('fps')}")
                st.write(f"**Codec:** {details.get('codec_name')}")
                if "ffprobe" in details:
                    if ("ffprobe_error" in details["ffprobe"]):
                        st.error(details["ffprobe"]["ffprobe_error"]); st.info("Tip: Install ffprobe (ffmpeg) for richer video metadata.")
                    else:
                        with st.expander("Raw ffprobe JSON"):
                            st.code(json.dumps(details["ffprobe"], indent=2), language="json")
            else:
                st.info("File type not recognized as image/video. Raw details:"); st.json(details)
            st.divider()
            with st.expander("Full Forensic Summary (raw JSON)"):
                st.code(json.dumps(summary, indent=2), language="json")

# --------- Detection Tab (main + advanced inline) ---------
with tab_detect:
    st.subheader("Detection & Tracking")

    # Simple controls
    f_listing = api_forensics_files(); all_files: List[str] = f_listing.get("files", [])
    video_exts = {".mp4",".mov",".mkv",".avi",".m4v",".webm"}
    video_files = [f for f in all_files if Path(f).suffix.lower() in video_exts]
    left, right = st.columns([2, 3])

    with left:
        if not video_files: st.info("Upload a video in the **Upload & Hash** tab first.")
        selected_video = st.selectbox("Select a video (stored filename)", ["-- select --"] + video_files, index=0, key="detect_video_sel")
        defaults = api_detect_options()
        engine = st.selectbox("Engine", options=["motion","yolov8"], index=0 if defaults.get("engine","motion")=="motion" else 1, key="detect_engine_sel")
        sample_fps = st.slider("Sample FPS", 0.1, 30.0, float(defaults.get("sample_fps",2.0)), 0.1, key="detect_samplefps")
        max_frames = st.number_input("Max frames (None = all)", min_value=1, value=int(defaults.get("max_frames",200)), key="detect_maxframes")
        if engine == "motion":
            min_area = st.number_input("Min area (motion contour)", min_value=10, value=int(defaults.get("min_area",500)), key="detect_minarea")
            conf_thresh = None; iou_thresh_track = None
        else:
            st.info("YOLOv8 selected — ensure ultralytics + torch are installed and server restarted.")
            conf_thresh = st.slider("(YOLO) Confidence threshold", 0.01, 0.95, float(defaults.get("conf_thresh",0.25)), 0.01, key="detect_conf")
            iou_thresh_track = st.slider("(YOLO) IOU for track association", 0.1, 0.9, float(defaults.get("iou_thresh_track",0.4)), 0.05, key="detect_iou")
            min_area = None
        auto_add_synth = st.checkbox("Auto-add synthesized findings to Report", value=True, key="detect_auto_add_synth")
        auto_candidates = st.checkbox("Auto-add detections/tracks as candidates (with thumbnails)", value=True, key="detect_auto_candidates")

        run_btn = st.button("Run detection", type="primary", disabled=(selected_video=="-- select --"), key="detect_run_btn")
        if run_btn:
            params = {"engine": engine, "sample_fps": float(sample_fps), "max_frames": int(max_frames)}
            if engine == "motion": params["min_area"] = int(min_area if min_area is not None else 500)
            else:
                params["conf_thresh"] = float(conf_thresh if conf_thresh is not None else 0.25)
                params["iou_thresh_track"] = float(iou_thresh_track if iou_thresh_track is not None else 0.4)
            result = api_detect_by_filename(selected_video, params)
            st.session_state["_detect_result"] = result; st.session_state["_detect_file"] = selected_video
            st.success(f"Detection complete. Engine: {result.get('meta',{}).get('engine','motion')}")
            if auto_add_synth:
                try:
                    added = add_synth_findings_to_report_session(result, selected_video)
                    if added>0:
                        st.success(f"Synthesized and added {added} finding(s) to Report.") 
                    else: 
                        st.info("No findings could be synthesized from this run.")
                except Exception as e:
                    st.warning(f"Auto-add (synth) failed: {e}")
            if auto_candidates:
                try:
                    c = add_candidates_from_detection_result(result, selected_video, make_thumbs=True)
                    if c>0: st.success(f"Added {c} candidate(s) to Report → Analysis Findings.")
                except Exception as e:
                    st.warning(f"Auto-candidate failed: {e}")

    with right:
        result = st.session_state.get("_detect_result"); sel_file = st.session_state.get("_detect_file")
        if result and sel_file:
            meta = result.get("meta", {}); dets = result.get("detections", []); tracks = result.get("tracks", [])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Processed frames", f"{meta.get('processed_frames', 0)}")
            m2.metric("Detections", f"{len(dets)}")
            m3.metric("Tracks", f"{len(tracks)}")
            m4.metric("Sample FPS", f"{meta.get('fps_sampling', 0)}")

            video_path = meta.get("video_path")
            if not video_path or not Path(video_path).exists():
                st.warning("Video path not accessible for preview. (Still fine; JSON below)")
            else:
                det_by_frame = group_detections_by_frame(dets); frames_available = sorted(det_by_frame.keys())
                if frames_available:
                    frame_choice = st.slider("Preview frame index", min_value=frames_available[0], max_value=frames_available[-1],
                                             value=frames_available[0], step=1, key="detect_preview_frame")
                    img_prev = draw_overlays_on_frame(video_path, frame_choice, det_by_frame.get(frame_choice, []))
                    st.image(img_prev, caption=f"Frame {frame_choice} with overlays", use_container_width=True)
                else: st.info("No detections found on sampled frames.")
            st.divider() 
            st.markdown("### Track Events") 
            if tracks:
                st.json(tracks)
            else: 
                st.write("No tracks created.")
            st.divider()
            st.markdown("### Raw Detection JSON")
            with st.expander("Show JSON"): st.code(json.dumps(result, indent=2), language="json")
            st.download_button(label="Download detection JSON", data=json.dumps(result, indent=2),
                               file_name=f"{Path(sel_file).stem}_detection.json", mime="application/json", key="detect_download_json")

    # Advanced Detection & Tracking inline (stored uploads)
    st.markdown("---")
    with st.expander("Advanced Detection & Tracking (Stored uploads: data/uploads)", expanded=True):
        API_BASE_API = _api_base_api()
        def _post_api(url_tail: str, data: Dict[str, Any], files: Dict[str, Any] | None = None) -> Dict[str, Any]:
            url = f"{API_BASE_API}{url_tail}"
            r = requests.post(url, data=data, files=files, timeout=1200)
            if not r.ok: raise RuntimeError(f"{url} failed ({r.status_code}): {r.text}")
            return r.json()
        def _norm_path(p: str | None) -> str | None:
            if not p: return None
            pp = Path(p); pp = pp if pp.is_absolute() else (PROJECT_ROOT / p).resolve()
            return str(pp)
        def _media_preview(path_like: str | None, title: str):
            st.markdown(f"**{title}**")
            if not path_like: st.info("No output path."); return
            abspath = _norm_path(path_like)
            if not abspath or not Path(abspath).exists(): st.warning(f"Not found: {path_like}"); return
            suf = Path(abspath).suffix.lower(); st.caption(f"Path: `{abspath}` • exists: **{Path(abspath).exists()}**")
            if suf in {".jpg",".jpeg",".png",".bmp",".webp"}: st.image(abspath, use_container_width=True)
            elif suf in {".mp4",".mov",".m4v",".webm"}:
                try: st.video(abspath)
                except Exception:
                    with open(abspath, "rb") as f: st.video(f.read())
            else: st.write(f"Saved: `{abspath}`")

        auto_add_findings = st.checkbox("Auto-add backend findings to Report", value=True, key="adv_auto_add_findings")
        auto_candidates_adv = st.checkbox("Auto-add detections/tracks as candidates (with thumbnails)", value=True, key="adv_auto_cand")

        stored_files = [f for f in all_files]
        imgs = [f for f in stored_files if Path(f).suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}]
        vids = [f for f in stored_files if Path(f).suffix.lower() in {".mp4",".mov",".m4v",".webm",".mkv",".avi"}]

        sub = st.tabs(["Detect (Image, stored)", "Detect (Video, stored)", "Track (Video, stored)"])

        with sub[0]:
            pick = st.selectbox("Stored image", ["-- select --"] + imgs, index=0, key="adv_det_img_pick")
            colA, colB, colC = st.columns(3)
            conf = colA.slider("Confidence", 0.1, 0.9, 0.35, 0.01, key="adv_det_img_conf")
            iou  = colB.slider("NMS IoU", 0.1, 0.9, 0.50, 0.01, key="adv_det_img_iou")
            model = colC.selectbox("Model", ["yolov8l.pt","yolov8x.pt"], index=0, key="adv_det_img_model")
            if st.button("Run detection (by-filename)", disabled=(pick=="-- select --"), key="adv_det_img_run"):
                res = _post_api(f"/detect/image/by-filename/{pick}", data={"model_name": model, "conf": conf, "iou": iou})
                st.json(res); _media_preview(res.get("annotated_path"), "Annotated image")
                # backend findings
                if auto_add_findings:
                    added0 = 0; fj = res.get("findings_json_path")
                    if fj:
                        try:
                            p = Path(_norm_path(fj)); 
                            if p and p.exists():
                                arr = json.loads(p.read_text(encoding="utf-8")).get("findings", [])
                                if isinstance(arr, list) and arr:
                                    st.session_state.setdefault("rep_findings", []).extend(arr); added0 += len(arr)
                        except Exception as e: st.warning(f"Auto-add: could not read findings_json_path: {e}")
                    if added0>0: st.success(f"Auto-added {added0} finding(s) to Report.")
                # candidates from detections (image)
                if auto_candidates_adv:
                    try:
                        c = add_candidates_from_detection_result(res, pick, make_thumbs=True)
                        if c>0: st.success(f"Added {c} candidate(s) to Report → Analysis Findings.")
                    except Exception as e: st.warning(f"Auto-candidate failed: {e}")

        with sub[1]:
            pick = st.selectbox("Stored video", ["-- select --"] + vids, index=0, key="adv_det_vid_pick")
            colA, colB, colC, colD = st.columns(4)
            conf = colA.slider("Confidence", 0.1, 0.9, 0.35, 0.01, key="adv_det_vid_conf")
            iou  = colB.slider("NMS IoU", 0.1, 0.9, 0.50, 0.01, key="adv_det_vid_iou")
            stride = colC.number_input("Frame stride", 1, 8, 1, step=1, key="adv_det_vid_stride")
            maxf   = colD.number_input("Max frames", 0, 5000, 0, step=50, key="adv_det_vid_maxf", help="0 = no cap")
            model = st.selectbox("Model", ["yolov8l.pt","yolov8x.pt"], index=0, key="adv_det_vid_model")
            if st.button("Run detection (by-filename)", disabled=(pick=="-- select --"), key="adv_det_vid_run"):
                data = {"model_name": model, "conf": conf, "iou": iou, "stride": int(stride), "max_frames": None if int(maxf)==0 else int(maxf)}
                res = _post_api(f"/detect/video/by-filename/{pick}", data=data)
                st.json(res); _media_preview(res.get("annotated_path"), "Annotated video")
                if auto_add_findings:
                    added0 = 0; fj = res.get("findings_json_path")
                    if fj:
                        try:
                            p = Path(_norm_path(fj)); 
                            if p and p.exists():
                                arr = json.loads(p.read_text(encoding="utf-8")).get("findings", [])
                                if isinstance(arr, list) and arr:
                                    st.session_state.setdefault("rep_findings", []).extend(arr); added0 += len(arr)
                        except Exception as e: st.warning(f"Auto-add: could not read findings_json_path: {e}")
                    if added0>0: st.success(f"Auto-added {added0} finding(s) to Report.")
                if auto_candidates_adv:
                    try:
                        c = add_candidates_from_detection_result(res, pick, make_thumbs=True)
                        if c>0: st.success(f"Added {c} candidate(s) to Report → Analysis Findings.")
                    except Exception as e: st.warning(f"Auto-candidate failed: {e}")

        with sub[2]:
            pick = st.selectbox("Stored video", ["-- select --"] + vids, index=0, key="adv_trk_vid_pick")
            colA, colB, colC = st.columns(3)
            conf = colA.slider("Confidence", 0.1, 0.9, 0.35, 0.01, key="adv_trk_conf")
            iou  = colB.slider("NMS IoU", 0.1, 0.9, 0.50, 0.01, key="adv_trk_iou")
            minlen = colC.number_input("Min track length (frames)", 1, 100, 5, step=1, key="adv_trk_minlen")
            model = st.selectbox("Model", ["yolov8l.pt","yolov8x.pt"], index=0, key="adv_trk_model")
            device = st.selectbox("Device", ["cpu"], index=0, key="adv_trk_device")
            if st.button("Run tracking (by-filename)", disabled=(pick=="-- select --"), key="adv_trk_run"):
                data = {"model_name": model, "conf": conf, "iou": iou, "min_track_len": int(minlen), "device": None if device=="cpu" else device}
                res = _post_api(f"/track/video/by-filename/{pick}", data=data)
                st.json(res); _media_preview(res.get("annotated_path"), "Annotated tracking video")
                if auto_add_findings:
                    added0 = 0; fj = res.get("findings_json_path")
                    if fj:
                        try:
                            p = Path(_norm_path(fj)); 
                            if p and p.exists():
                                arr = json.loads(p.read_text(encoding="utf-8")).get("findings", [])
                                if isinstance(arr, list) and arr:
                                    st.session_state.setdefault("rep_findings", []).extend(arr); added0 += len(arr)
                        except Exception as e: st.warning(f"Auto-add: could not read findings_json_path: {e}")
                    if added0>0: st.success(f"Auto-added {added0} finding(s) to Report.")
                if auto_candidates_adv:
                    try:
                        c = add_candidates_from_detection_result(res, pick, make_thumbs=True)
                        if c>0: st.success(f"Added {c} candidate(s) to Report → Analysis Findings.")
                    except Exception as e: st.warning(f"Auto-candidate failed: {e}")

# --------- Faces Tab ---------
with tab_faces:
    st.subheader("Faces (Index & Search)")
    lcol, rcol = st.columns([2, 3])
    st.session_state.setdefault("_auto_face_candidates", True)
    st.checkbox("Auto-add face search matches as candidates", value=st.session_state["_auto_face_candidates"], key="_auto_face_candidates")

    with lcol:
        st.markdown("**Index from stored video**")
        f_listing = api_forensics_files(); filenames: List[str] = f_listing.get("files", [])
        vids = [f for f in filenames if Path(f).suffix.lower() in {".mp4",".mov",".mkv",".avi",".m4v",".webm"}]
        pick = st.selectbox("Video filename", ["-- select --"] + vids, index=0, key="faces_pick_vid")
        sfps = st.slider("Sample FPS", 0.1, 10.0, 1.0, 0.1, key="faces_sfps")
        mframes = st.number_input("Max frames", 10, 5000, 300, step=10, key="faces_mframes")
        mfaces = st.number_input("Max faces to index", 1, 5000, 200, step=10, key="faces_mfaces")
        cols = st.columns(3)
        if cols[0].button("Extract faces", disabled=(pick=="-- select --"), key="faces_extract"):
            try: res = api_faces_extract_by_filename(pick, sfps, mframes); st.success("Face extraction complete."); st.json(res)
            except requests.HTTPError as e: st.error(f"Extract failed ({e.response.status_code}): {e.response.text if e.response else ''}")
        if cols[1].button("Reset index", key="faces_reset"):
            res = api_faces_index_reset(); st.success("Index reset."); st.json(res)
        if cols[2].button("Index faces (add)", disabled=(pick=="-- select --"), key="faces_index_add"):
            try: res = api_faces_index_add_by_filename(pick, sfps, mframes, mfaces); st.success("Index updated."); st.json(res)
            except requests.HTTPError as e: st.error(f"Index add failed ({e.response.status_code}): {e.response.text if e.response else ''}")

        st.markdown("**Quick image search**")
        qimg = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"], key="faces_query_img")
        topk = st.number_input("Top-K", 1, 50, 5, step=1, key="faces_topk")
        if st.button("Search by image", disabled=(qimg is None), key="faces_search_img"):
            try:
                res = api_faces_search(qimg.getvalue(), qimg.name, int(topk))
                st.session_state["_face_img_search"] = res
                st.success("Search completed. See results on the right.")
        # NEW: auto-add as report candidates (toggle default ON)
                auto_face_candidates = st.session_state.get("_auto_face_candidates", True)
                if auto_face_candidates:
                    items = res.get("matches", []) or res.get("results", []) or []
                    if items:
                        added = add_candidates_from_face_matches(items, make_thumbs=True, max_items=min(len(items), 25))
                        if added > 0:
                            st.success(f"Auto-added {added} face candidate(s) to Report → Analysis Findings.")
            except requests.HTTPError as e:
                st.error(f"Search failed ({e.response.status_code}). {e.response.text if e.response else ''}")


    with rcol:
        st.markdown("**Search by embedding (from detected face)**")
        if st.button("Refresh index stats", key="faces_stats_btn"):
            try: st.session_state["_face_index_stats"] = api_faces_index_stats()
            except Exception as e: st.warning(f"Could not read index stats: {e}")
        st.json(st.session_state.get("_face_index_stats", {}))

        res = st.session_state.get("_face_img_search")
        if res:
            items = res.get("matches", []) or res.get("results", []) or []
            st.markdown(f"**Results:** {len(items)}")
            added_candidates = st.checkbox("Add matches as report candidates (with thumbnails)", value=True, key="faces_add_as_candidates")
            for i, it in enumerate(items):
                cols = st.columns([2, 2, 2])
                src = it.get("source_file") or it.get("file") or ""
                bbox = it.get("bbox"); frame_idx = it.get("frame_idx")
                if src and bbox:
                    try:
                        thumb = crop_from_source(src, bbox, frame_idx)
                        if thumb is not None: cols[0].image(thumb, caption=f"{src}", use_container_width=True)
                    except Exception: pass
                cols[1].write(it)
                embedding = it.get("normed_embedding") or it.get("embedding")
                if embedding:
                    st.caption(f"Embedding length: {len(embedding)}")
                if added_candidates and cols[2].button(f"Add as candidate #{i+1}", key=f"add_cand_face_{i}"):
                    try:
                        added = add_candidates_from_face_matches([it], default_source=src, make_thumbs=True, max_items=1)
                        st.success(f"Added {added} face candidate(s).")
                    except Exception as e:
                        st.warning(f"Could not add candidate: {e}")

# --------- Report & ELA Tab ---------
with tab_report:
    st.subheader("Report & ELA")
    st.session_state.setdefault("rep_candidates", [])
    st.info(f"Candidates in session: **{len(st.session_state['rep_candidates'])}**  |  Manual findings: **{len(st.session_state.get('rep_findings', []))}**")

    # ELA
    listing = api_forensics_files(); all_files: List[str] = listing.get("files", [])
    image_exts = {".jpg",".jpeg",".png",".bmp",".webp"}; video_exts = {".mp4",".mov",".mkv",".avi",".m4v",".webm"}

    st.markdown("### ELA (Error Level Analysis)")
    col_ela_l, col_ela_r = st.columns([2, 3])
    with col_ela_l:
        pick = st.selectbox("Stored file (image or video)", options=["-- select --"] + all_files, index=0, key="ela_pick")
        jpeg_quality = st.slider("JPEG quality (recompress)", 10, 95, 90, 1, key="ela_q")
        hi_thresh = st.slider("Highlight threshold (0-255)", 0, 255, 40, 1, key="ela_thr")
        frame_idx = st.number_input("Frame index for video", min_value=0, value=0, step=1, key="ela_frame") if pick!="-- select --" and Path(pick).suffix.lower() in video_exts else None
        run_ela_btn = st.button("Run ELA", type="primary", key="ela_run")
    with col_ela_r:
        if run_ela_btn:
            try:
                res = api_ela_by_filename(pick, jpeg_quality=jpeg_quality, hi_thresh=hi_thresh, frame_idx=frame_idx)
                st.session_state["_ela_last"] = res; st.success("ELA generated.")
            except requests.HTTPError as e:
                st.error(f"ELA failed ({getattr(e.response,'status_code','HTTP')}): {e.response.text if e.response else ''}")
        ela_res = st.session_state.get("_ela_last")
        if ela_res:
            st.write("**Stats:**"); st.json({k: v for k, v in ela_res.items() if k not in {"original_path","ela_path"}})
            try:
                c1, c2 = st.columns(2)
                with c1: st.image(Image.open((PROJECT_ROOT / ela_res["original_path"]).resolve()), caption="Original", use_container_width=True)
                with c2: st.image(Image.open((PROJECT_ROOT / ela_res["ela_path"]).resolve()), caption="ELA heatmap", use_container_width=True)
            except Exception as e: st.warning(f"Could not preview ELA images: {e}")
            if st.button("Add this ELA image to report thumbnails", key="ela_add_thumb"):
                thumbs = st.session_state.get("_report_ela_thumbs", []); thumbs.append(ela_res["ela_path"])
                st.session_state["_report_ela_thumbs"] = thumbs; st.info(f"Added. Thumbs in report: {len(thumbs)}")

    st.divider()
    st.markdown("### Harvest candidates from saved JSON (optional)")
    harv_col1, harv_col2 = st.columns([3, 1])

# Scan known folders for JSONs
    det_json_dir = (PROJECT_ROOT / "data" / "detections")
    trk_json_dir = (PROJECT_ROOT / "data" / "tracks")
    json_files = []
    if det_json_dir.exists():
        json_files += [str(p.relative_to(PROJECT_ROOT)) for p in det_json_dir.glob("*.json")]
    if trk_json_dir.exists():
        json_files += [str(p.relative_to(PROJECT_ROOT)) for p in trk_json_dir.glob("*.json")]
    json_files = sorted(json_files)

    json_pick = harv_col1.multiselect("Select detection/tracking JSON files to harvest", options=json_files, default=[])
    go_harvest = harv_col2.button("Harvest", type="secondary")

    if go_harvest and json_pick:
        harvested = 0
        for rel in json_pick:
            blob = _load_json_file(rel)
            if not blob:
                continue
        # Try to infer the source filename from the blob; fall back to stem
            src = blob.get("meta", {}).get("source_file") or Path(rel).stem
            harvested += add_candidates_from_detection_result(blob, src, make_thumbs=True)
        st.success(f"Harvested {harvested} candidate(s).")

    # NEW: Candidates section
    st.markdown("### Analysis Findings (candidates)")
    st.caption("These are automatically collected from Detection/Tracking/Faces. Uncheck to exclude from the final report.")
    st.session_state.setdefault("rep_candidates", [])
    # quick controls
    c_cols = st.columns([1,1,1,1])
    if c_cols[0].button("Select all", key="cand_sel_all"):
        for c in st.session_state["rep_candidates"]: c["include"] = True
    if c_cols[1].button("Select none", key="cand_sel_none"):
        for c in st.session_state["rep_candidates"]: c["include"] = False
    if c_cols[2].button("Clear all candidates", key="cand_clear_all"):
        st.session_state["rep_candidates"] = []
    show_json = c_cols[3].checkbox("Show raw JSON", value=False, key="cand_show_json")

    # list candidates
    if not st.session_state["rep_candidates"]:
        st.info("No analysis candidates yet. Run Detection/Tracking or add from Faces.")
    else:
        for idx, c in enumerate(st.session_state["rep_candidates"]):
            box = st.container(border=True)
            col1, col2, col3 = box.columns([2,3,1])
            # thumbnail
            if c.get("snapshot_path"):
                p = (PROJECT_ROOT / c["snapshot_path"]).resolve()
                if p.exists():
                    col1.image(Image.open(p), caption=c.get("source_file",""), use_container_width=True)
                else:
                    col1.write("No snapshot.")
            else:
                col1.write("No snapshot.")
            # meta
            meta_str = f"**Type:** {c.get('type')} | **Label:** {c.get('label')} | **Conf:** {c.get('confidence')}\n\n"
            meta_str += f"**Source:** {c.get('source_file')} | **Frame:** {c.get('frame_idx')} | **Time:** {c.get('timestamp_sec') if c.get('timestamp_sec') is not None else '—'}\n\n"
            meta_str += f"**BBox:** {c.get('bbox')}"
            col2.markdown(meta_str)
            # include + remove
            c["include"] = col3.checkbox("Include", value=c.get("include", True), key=f"cand_inc_{idx}")
            if col3.button("Remove", key=f"cand_rm_{idx}"):
                st.session_state["rep_candidates"].pop(idx); st.rerun()
            if show_json:
                with box.expander("Raw candidate JSON"):
                    st.code(json.dumps(c, indent=2), language="json")

    st.divider()
    st.markdown("### Build PDF Report")

    # Header
    h_case_id = st.text_input("Case ID", key="rep_case_id")
    h_investigator = st.text_input("Investigator", key="rep_investigator")
    h_station = st.text_input("Station/Unit", key="rep_station")
    h_contact = st.text_input("Contact", key="rep_contact")
    h_notes = st.text_area("Case Notes", key="rep_notes")

    # Evidence
    ev_pick = st.multiselect("Select evidence files", options=all_files, default=[], key="rep_ev_pick")
    evidence_payload: List[Dict[str, Any]] = []
    for f in ev_pick:
        summ = api_forensics_by_filename(f); det = summ.get("details", {})
        dur = float(det.get("duration_sec", 0.0) or 0.0) if summ.get("kind") == "video" else None
        camera_id = st.text_input(f"Camera ID for {f}", value="unknown", key=f"rep_cam_{f}")
        evidence_payload.append({"filename": f, "sha256": summ.get("sha256",""), "ingest_time": datetime.datetime.utcnow().isoformat()+"Z", "camera_id": camera_id, "duration": dur})

    # Manual findings editor (existing)
    st.markdown("**Manual Findings (you can still add/edit)**")
    st.session_state.setdefault("rep_findings", [])
    if st.button("Add empty finding", key="rep_f_add"):
        st.session_state["rep_findings"].append({"time_window":"", "track_id":None, "object_type":"person",
                                                 "representative_frame_path":"", "bbox":None, "matched_offender_id":None,
                                                 "matched_offender_name":None, "similarity_score":None, "verification_status":"unverified"})
    new_list = []
    for idx, f in enumerate(st.session_state["rep_findings"]):
        st.markdown(f"**Event #{idx+1}**"); c1, c2 = st.columns(2)
        with c1:
            f["time_window"] = st.text_input("Time window", value=f.get("time_window",""), key=f"rep_f_tw_{idx}")
            tid_val = f.get("track_id")
            f["track_id"] = st.number_input("Track ID (optional)", min_value=0, value=int(tid_val or 0), step=1, key=f"rep_f_tid_{idx}") if tid_val is not None else None
            f["object_type"] = st.selectbox("Object type", ["person","car","motion","object"], index=["person","car","motion","object"].index(f.get("object_type","person")), key=f"rep_f_obj_{idx}")
            f["representative_frame_path"] = st.text_input("Representative image path (optional)", value=f.get("representative_frame_path",""), key=f"rep_f_repr_{idx}")
        with c2:
            f["matched_offender_id"] = st.text_input("Matched offender ID (optional)", value=f.get("matched_offender_id") or "", key=f"rep_f_oid_{idx}") or None
            f["matched_offender_name"] = st.text_input("Matched offender Name (optional)", value=f.get("matched_offender_name") or "", key=f"rep_f_oname_{idx}") or None
            sim = st.text_input("Similarity score (0..1, optional)", value="" if f.get("similarity_score") is None else str(f.get("similarity_score")), key=f"rep_f_sim_{idx}")
            f["similarity_score"] = float(sim) if sim.strip() else None
            f["verification_status"] = st.selectbox("Verification status", ["verified","unverified","unknown"], index=["verified","unverified","unknown"].index(f.get("verification_status","unverified")), key=f"rep_f_ver_{idx}")
        bx = st.text_input("BBox x,y,w,h (optional)", value=(",".join(map(str, f.get("bbox") or [])) if f.get("bbox") else ""), key=f"rep_f_bbox_{idx}")
        if bx.strip():
            try: x,y,w,h = [int(v.strip()) for v in bx.split(",")]; f["bbox"] = [x,y,w,h]
            except Exception: st.warning("Invalid bbox format; expected x,y,w,h"); f["bbox"] = None
        if st.button("Remove this event", key=f"rep_f_rm_{idx}"): pass
        else: new_list.append(f)
        st.markdown("---")
    st.session_state["rep_findings"] = new_list

    # Forensics
    st.markdown("**Forensics block**")
    meta_summary = api_forensics_by_filename(ev_pick[0]) if ev_pick else {}
    thumbs = st.session_state.get("_report_ela_thumbs", []); st.write(f"ELA thumbnails queued: **{len(thumbs)}**")
    deepfake = st.slider("Deepfake heuristic score (0..1; optional)", 0.0, 1.0, 0.0, 0.01, key="rep_df")

    # Build + send payload
    if st.button("Generate PDF Report", type="primary", key="rep_generate"):
        try:
            # merge manual findings + included candidates (mapped)
            candidate_findings = _candidates_to_report_findings(selected_only=True)
            all_findings = list(st.session_state["rep_findings"]) + candidate_findings
            payload = {"header":{"case_id":h_case_id,"investigator":h_investigator,"station_unit":h_station,"contact":h_contact,"case_notes":h_notes},
                       "evidence":evidence_payload,
                       "findings":all_findings,
                       "forensics":{"metadata_summary":meta_summary,"tamper_flags":[],"ela_thumbnails":thumbs,"deepfake_score":float(deepfake)},
                       "bundle_json":{}}
            out = api_report_generate(payload); st.success("Report generated.")
            st.json({k: v for k,v in out.items() if k not in {"pdf_path","json_path","qr_path"}})
            _abs = lambda p: str((PROJECT_ROOT / p).resolve())
            st.markdown(f"[Download PDF]({_abs(out['pdf_path'])})")
            st.markdown(f"[Download JSON bundle]({_abs(out['json_path'])})")
            st.markdown(f"[QR image]({_abs(out['qr_path'])})")
        except requests.HTTPError as e:
            st.error(f"Report generation failed ({getattr(e.response,'status_code','HTTP')}): {e.response.text if e.response else ''}")

# --------- Verify PDF Tab (right-most)
with tab_verify:
    st.subheader("Verify Report (PDF)")
    st.caption("Upload a generated PDF. Optionally add a public key (PEM) and/or the report’s JSON bundle for cross-check.")
    c1, c2 = st.columns([2,1])
    with c1:
        up_pdf = st.file_uploader("Report PDF", type=["pdf"], key="ver_pdf")
        up_pem = st.file_uploader("Public key (PEM, optional)", type=["pem"], key="ver_pem")
        up_json = st.file_uploader("Bundle JSON (optional)", type=["json"], key="ver_json")
        run = st.button("Run verification", type="primary", disabled=(up_pdf is None))
    with c2:
        st.write("**API_BASE**:", f"`{_api_base_api()}`")
    if run and up_pdf is not None:
        try:
            res = api_verify_pdf(up_pdf.getvalue(), up_pem.getvalue() if up_pem else None, up_json.getvalue() if up_json else None)
            ok = res.get("ok"); st.success("Verification PASSED") if ok else st.error("Verification FAILED")
            st.json(res)
            st.code(f"printed report_sha256: {res.get('printed_sha256')}\ncomputed sha256:     {res.get('computed_sha256')}", language="text")
        except requests.HTTPError as e:
            st.error(f"Verify failed ({getattr(e.response,'status_code','HTTP')}): {e.response.text if e.response else ''}")
##