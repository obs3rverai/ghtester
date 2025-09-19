# ui/dashboard.py
import io
import json
import mimetypes
from pathlib import Path
from typing import Dict, Any, List, Tuple, DefaultDict
from collections import defaultdict

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

# --- config ---
API_BASE = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="AI CCTV & Digital Media Forensic Tool — MVP", layout="wide")
st.title("AI CCTV & Digital Media Forensic Tool — MVP")
st.caption("Hackathon demo UI (uploads, forensics, motion/YOLO detection, faces index & search).")

# --- helpers (HTTP) ---
def api_health() -> dict:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": f"error: {e}"}

def api_ingest(file_bytes: bytes, filename: str) -> dict:
    files = {"file": (filename, file_bytes, mimetypes.guess_type(filename)[0] or "application/octet-stream")}
    r = requests.post(f"{API_BASE}/ingest", files=files, timeout=60)
    r.raise_for_status()
    return r.json()

def api_list_files() -> dict:
    r = requests.get(f"{API_BASE}/files", timeout=10)
    r.raise_for_status()
    return r.json()

def api_forensics_files() -> dict:
    r = requests.get(f"{API_BASE}/forensics/files", timeout=10)
    r.raise_for_status()
    return r.json()

def api_forensics_by_filename(stored_name: str) -> dict:
    r = requests.get(f"{API_BASE}/forensics/summary/by-filename/{stored_name}", timeout=60)
    r.raise_for_status()
    return r.json()

def api_detect_options() -> dict:
    r = requests.get(f"{API_BASE}/detect/options", timeout=10)
    r.raise_for_status()
    return r.json()

def api_detect_by_filename(stored_name: str, params: dict) -> dict:
    r = requests.post(f"{API_BASE}/detect/run/by-filename/{stored_name}", json=params, timeout=1200)
    r.raise_for_status()
    return r.json()

# Faces APIs
def api_faces_extract_by_filename(stored_name: str, sample_fps: float, max_frames: int) -> dict:
    r = requests.post(f"{API_BASE}/faces/extract/by-filename/{stored_name}",
                      json={"sample_fps": float(sample_fps), "max_frames": int(max_frames)}, timeout=600)
    r.raise_for_status()
    return r.json()

def api_faces_index_reset() -> dict:
    r = requests.post(f"{API_BASE}/faces/index/reset", timeout=10)
    r.raise_for_status()
    return r.json()

def api_faces_index_add_by_filename(stored_name: str, sample_fps: float, max_frames: int, max_faces: int) -> dict:
    r = requests.post(f"{API_BASE}/faces/index/add/by-filename/{stored_name}",
                      json={"sample_fps": float(sample_fps), "max_frames": int(max_frames), "max_faces": int(max_faces)},
                      timeout=1200)
    r.raise_for_status()
    return r.json()

def api_faces_index_stats() -> dict:
    r = requests.get(f"{API_BASE}/faces/index/stats", timeout=10)
    r.raise_for_status()
    return r.json()

def api_faces_search(file_bytes: bytes, filename: str, top_k: int) -> dict:
    files = {"file": (filename, file_bytes, mimetypes.guess_type(filename)[0] or "application/octet-stream")}
    data = {"top_k": str(int(top_k))}
    r = requests.post(f"{API_BASE}/faces/search", files=files, data=data, timeout=120)
    r.raise_for_status()
    return r.json()

# --- helpers (local preview) ---
def try_show_media(stored_path: str):
    abs_path = (PROJECT_ROOT / stored_path).resolve()
    if not abs_path.exists():
        st.info("File saved, but preview not found on this path.")
        return
    suffix = abs_path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        try:
            img = Image.open(abs_path)
            st.image(img, caption=abs_path.name, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not open image: {e}")
    elif suffix in {".mp4", ".mov", ".mkv", ".avi"}:
        try:
            st.video(str(abs_path))
        except Exception as e:
            st.warning(f"Could not play video: {e}")
    else:
        st.write("Preview not supported for this file type.")

def draw_overlays_on_frame(video_path: str, frame_idx: int, dets_this_frame: List[Dict[str, Any]]) -> Image.Image:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for preview: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    for d in dets_this_frame:
        x, y, w, h = d["bbox"]
        x2, y2 = x + w, y + h
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        label = f"{d.get('label','obj')} {d.get('conf', 0):.2f}"
        cv2.putText(frame, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def group_detections_by_frame(detections: List[Dict[str, Any]]) -> DefaultDict[int, List[Dict[str, Any]]]:
    grouped: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for d in detections:
        grouped[int(d["frame_idx"])].append(d)
    return grouped

def crop_from_source(source_file: str, bbox: List[int], frame_idx: int | None) -> Image.Image | None:
    """Return a PIL image crop for a match from image or video source."""
    src_path = (PROJECT_ROOT / "data" / "uploads" / source_file).resolve()
    if not src_path.exists():
        return None
    x, y, w, h = map(int, bbox)
    if src_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        img = cv2.imread(str(src_path))
        if img is None:
            return None
        H, W = img.shape[:2]
        x2, y2 = min(x + w, W - 1), min(y + h, H - 1)
        crop = img[max(0, y):y2, max(0, x):x2, :]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    else:
        # video frame grab
        if frame_idx is None:
            return None
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return None
        H, W = frame.shape[:2]
        x2, y2 = min(x + w, W - 1), min(y + h, H - 1)
        crop = frame[max(0, y):y2, max(0, x):x2, :]
        return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

# ============== HEADER ROWS ==============
health_col, files_col = st.columns([1, 2])
with health_col:
    st.subheader("Server Health")
    health = api_health()
    yolo_avail = health.get("yolo_available", "no")
    if health.get("status") == "ok":
        st.success(f"Backend is running ✅  |  YOLO: {yolo_avail}")
    else:
        st.error(f"Backend issue: {health.get('status')}")

with files_col:
    st.subheader("Stored Files")
    if st.button("Refresh list"):
        pass
    try:
        listing = api_list_files()
        st.write(f"Count: **{listing.get('count', 0)}**")
        files = listing.get("files", [])
        if files:
            st.code("\n".join(files), language="text")
        else:
            st.info("No files yet. Upload one below.")
    except Exception as e:
        st.error(f"Could not fetch file list: {e}")

st.markdown("---")

# ============== TABS ==============
tab_upload, tab_forensics, tab_detect, tab_faces = st.tabs(
    ["📤 Upload & Hash", "🧪 Forensics Metadata", "🎯 Detection", "👤 Faces (Index & Search)"]
)

# --------- Upload Tab ---------
with tab_upload:
    st.subheader("Upload & Hash")
    uploaded = st.file_uploader(
        "Select an image/video to ingest",
        type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "mov", "mkv", "avi"]
    )
    if uploaded is not None:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.write("**Selected file:**", uploaded.name)
            if uploaded.type.startswith("image/"):
                try:
                    st.image(uploaded, caption="Preview", use_container_width=True)
                except Exception:
                    pass
            elif uploaded.type.startswith("video/"):
                st.info("Video selected (preview after upload).")
        if st.button("Upload & Compute SHA-256", type="primary"):
            try:
                resp = api_ingest(uploaded.getvalue(), uploaded.name)
                st.success("Uploaded successfully!")
                st.json(resp)
                st.markdown("**Local Preview (from stored path):**")
                try_show_media(resp.get("stored_path", ""))
            except requests.HTTPError as e:
                try:
                    st.error(f"Upload failed: {e.response.text}")
                except Exception:
                    st.error(f"Upload failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# --------- Forensics Tab ---------
with tab_forensics:
    st.subheader("Forensic Metadata Viewer")
    left, right = st.columns([2, 3])
    with left:
        st.write("Pick a stored file to view EXIF/FFprobe + hashes.")
        try:
            f_listing = api_forensics_files()
            filenames: List[str] = f_listing.get("files", [])
        except Exception as e:
            filenames = []
            st.error(f"Could not fetch list: {e}")
        selected = st.selectbox(
    "Stored filename",
    options=["-- select --"] + filenames,
    index=0,
    key="forensics_file_sel",
)
        if st.button("Load metadata", type="primary", disabled=(selected == "-- select --")):
            st.session_state["_selected_forensics_file"] = selected
    with right:
        sel = st.session_state.get("_selected_forensics_file")
        if sel and sel != "-- select --":
            try:
                summary = api_forensics_by_filename(sel)
                st.markdown(f"**File:** `{sel}`")
                base_info_cols = st.columns(4)
                base_info_cols[0].metric("Type", summary.get("kind"))
                base_info_cols[1].metric("MIME", summary.get("mime"))
                size_b = summary.get("size_bytes", 0)
                base_info_cols[2].metric("Size (bytes)", f"{size_b:,}")
                base_info_cols[3].markdown(f"`SHA-256`:\n\n`{summary.get('sha256')}`")
                st.divider()
                rel_path = summary.get("path")
                if rel_path:
                    try_show_media(Path(rel_path).relative_to(PROJECT_ROOT) if rel_path.startswith(str(PROJECT_ROOT)) else rel_path)
                details = summary.get("details", {})
                kind = summary.get("kind")
                if kind == "image":
                    st.markdown("### Image Details")
                    img_cols = st.columns(3)
                    img_cols[0].write(f"**Width:** {details.get('width')}")
                    img_cols[1].write(f"**Height:** {details.get('height')}")
                    img_cols[2].write(f"**Format/Mode:** {details.get('format')}/{details.get('mode')}")
                    exif = details.get("exif", {})
                    if exif:
                        st.markdown("#### EXIF")
                        st.json(exif)
                    if "error" in details:
                        st.warning(details["error"])
                elif kind == "video":
                    st.markdown("### Video Details")
                    v_cols = st.columns(4)
                    v_cols[0].write(f"**Duration (s):** {details.get('duration_sec')}")
                    v_cols[1].write(f"**Bitrate:** {details.get('bit_rate')}")
                    v_cols[2].write(f"**Resolution:** {details.get('width')}×{details.get('height')}")
                    v_cols[3].write(f"**FPS:** {details.get('fps')}")
                    st.write(f"**Codec:** {details.get('codec_name')}")
                    if "ffprobe" in details:
                        if "ffprobe_error" in details["ffprobe"]:
                            st.error(details["ffprobe"]["ffprobe_error"])
                            st.info("Tip: Install ffprobe (ffmpeg) for richer video metadata.")
                        else:
                            with st.expander("Raw ffprobe JSON"):
                                st.code(json.dumps(details["ffprobe"], indent=2), language="json")
                else:
                    st.info("File type not recognized as image/video. Raw details:")
                    st.json(details)
                st.divider()
                with st.expander("Full Forensic Summary (raw JSON)"):
                    st.code(json.dumps(summary, indent=2), language="json")
            except requests.HTTPError as e:
                try:
                    st.error(f"Metadata fetch failed: {e.response.text}")
                except Exception:
                    st.error(f"Metadata fetch failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# --------- Detection Tab ---------
with tab_detect:
    st.subheader("Detection & Tracking")
    try:
        f_listing = api_forensics_files()
        all_files: List[str] = f_listing.get("files", [])
    except Exception as e:
        all_files = []
        st.error(f"Could not fetch files: {e}")
    video_exts = {".mp4", ".mov", ".mkv", ".avi"}
    video_files = [f for f in all_files if Path(f).suffix.lower() in video_exts]
    left, right = st.columns([2, 3])
    with left:
        if not video_files:
            st.info("Upload a video in the **Upload & Hash** tab first.")
        selected_video = st.selectbox(
    "Select a video (stored filename)",
    ["-- select --"] + video_files,
    index=0,
    key="detect_video_sel",
)
        try:
            defaults = api_detect_options()
        except Exception:
            defaults = {"engine": "motion", "sample_fps": 2.0, "max_frames": 200, "min_area": 500, "conf_thresh": 0.25, "iou_thresh_track": 0.4}
        engine = st.selectbox(
    "Engine",
    options=["motion", "yolov8"],
    index=0 if defaults.get("engine","motion")=="motion" else 1,
    key="detect_engine_sel",
)
        sample_fps = st.slider("Sample FPS", 0.1, 30.0, float(defaults.get("sample_fps", 2.0)), 0.1)
        max_frames = st.number_input("Max frames (None = all)", min_value=1, value=int(defaults.get("max_frames", 200)))
        if engine == "motion":
            min_area = st.number_input("Min area (motion contour)", min_value=10, value=int(defaults.get("min_area", 500)))
            conf_thresh = None; iou_thresh_track = None
        else:
            st.info("YOLOv8 selected — ensure ultralytics + torch are installed and server restarted.")
            conf_thresh = st.slider("(YOLO) Confidence threshold", 0.01, 0.95, float(defaults.get("conf_thresh", 0.25)), 0.01)
            iou_thresh_track = st.slider("(YOLO) IOU for track association", 0.1, 0.9, float(defaults.get("iou_thresh_track", 0.4)), 0.05)
            min_area = None
        run_btn = st.button("Run detection", type="primary", disabled=(selected_video == "-- select --"))
        if run_btn:
            try:
                params = {"engine": engine, "sample_fps": float(sample_fps), "max_frames": int(max_frames)}
                if engine == "motion":
                    params["min_area"] = int(min_area if min_area is not None else 500)
                else:
                    params["conf_thresh"] = float(conf_thresh if conf_thresh is not None else 0.25)
                    params["iou_thresh_track"] = float(iou_thresh_track if iou_thresh_track is not None else 0.4)
                result = api_detect_by_filename(selected_video, params)
                st.session_state["_detect_result"] = result
                st.session_state["_detect_file"] = selected_video
                st.success(f"Detection complete. Engine: {result.get('meta',{}).get('engine','motion')}")
            except requests.HTTPError as e:
                try:
                    st.error(f"Detection failed: {e.response.text}")
                except Exception:
                    st.error(f"Detection failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
    with right:
        result = st.session_state.get("_detect_result")
        sel_file = st.session_state.get("_detect_file")
        if result and sel_file:
            meta = result.get("meta", {})
            dets = result.get("detections", [])
            tracks = result.get("tracks", [])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Processed frames", f"{meta.get('processed_frames', 0)}")
            m2.metric("Detections", f"{len(dets)}")
            m3.metric("Tracks", f"{len(tracks)}")
            m4.metric("Sample FPS", f"{meta.get('fps_sampling', 0)}")
            video_path = meta.get("video_path")
            if not video_path or not Path(video_path).exists():
                st.warning("Video path not accessible for preview. (Still fine; JSON below)")
            else:
                det_by_frame = group_detections_by_frame(dets)
                frames_available = sorted(det_by_frame.keys())
                if frames_available:
                    frame_choice = st.slider("Preview frame index", min_value=frames_available[0],
                                             max_value=frames_available[-1],
                                             value=frames_available[0], step=1)
                    try:
                        img_preview = draw_overlays_on_frame(video_path, frame_choice, det_by_frame.get(frame_choice, []))
                        st.image(img_preview, caption=f"Frame {frame_choice} with overlays", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not render frame preview: {e}")
                else:
                    st.info("No detections found on sampled frames.")
            st.divider()
            st.markdown("### Track Events")
            if tracks:
                st.json(tracks)
            else:
                st.write("No tracks created.")
            st.divider()
            st.markdown("### Raw Detection JSON")
            with st.expander("Show JSON"):
                st.code(json.dumps(result, indent=2), language="json")
            st.download_button(
                label="Download detection JSON",
                data=json.dumps(result, indent=2),
                file_name=f"{Path(sel_file).stem}_detection.json",
                mime="application/json"
            )

# --------- Faces Tab ---------
with tab_faces:
    st.subheader("Faces: Build Index & Search")

    try:
        f_listing = api_forensics_files()
        all_files: List[str] = f_listing.get("files", [])
    except Exception as e:
        all_files = []
        st.error(f"Could not fetch files: {e}")

    left, right = st.columns([2, 3])

    with left:
        st.markdown("#### 1) Index Builder")
        st.caption("Add faces from a stored image/video to the in-memory index.")
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        video_exts = {".mp4", ".mov", ".mkv", ".avi"}
        faceable_files = [f for f in all_files if Path(f).suffix.lower() in image_exts.union(video_exts)]

        idx_file = st.selectbox(
    "Stored filename",
    options=["-- select --"] + faceable_files,
    index=0,
    key="faces_index_sel",
)
        sample_fps = st.slider("Sample FPS (video only)", 0.1, 10.0, 1.0, 0.1)
        max_frames = st.number_input("Max frames to sample (video)", min_value=1, value=50)
        max_faces = st.number_input("Max faces to add", min_value=1, max_value=10000, value=500)

        col_btn1, col_btn2 = st.columns(2)
        if col_btn1.button("Add faces to index", type="primary", disabled=(idx_file == "-- select --")):
            try:
                resp = api_faces_index_add_by_filename(idx_file, sample_fps=float(sample_fps),
                                                       max_frames=int(max_frames), max_faces=int(max_faces))
                st.success(f"Added: {resp.get('added',0)}  |  Total in index: {resp.get('total',0)}")
                st.session_state["_face_index_stats"] = api_faces_index_stats()
            except requests.HTTPError as e:
                try:
                    st.error(f"Index add failed: {e.response.text}")
                except Exception:
                    st.error(f"Index add failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

        if col_btn2.button("Reset index", type="secondary"):
            try:
                api_faces_index_reset()
                st.session_state["_face_index_stats"] = {"count": 0}
                st.info("Face index reset.")
            except Exception as e:
                st.error(f"Reset failed: {e}")

        stats = st.session_state.get("_face_index_stats")
        if not stats:
            try:
                stats = api_faces_index_stats()
                st.session_state["_face_index_stats"] = stats
            except Exception:
                stats = {"count": 0}
        st.metric("Index size (faces)", stats.get("count", 0))

        st.markdown("---")
        st.markdown("#### 2) Extract (Preview Only, optional)")
        st.caption("See detected faces from a stored file (does not change index).")
        prev_file = st.selectbox("Pick file to preview faces", options=["-- select --"] + faceable_files, index=0, key="faces_preview_sel")
        if st.button("Preview detected faces"):
            if prev_file == "-- select --":
                st.warning("Pick a stored file first.")
            else:
                try:
                    out = api_faces_extract_by_filename(prev_file, sample_fps=float(sample_fps), max_frames=int(max_frames))
                    faces = out.get("faces", [])
                    st.write(f"Found faces: **{len(faces)}**")
                    # show up to 8 crops
                    for i, f in enumerate(faces[:8]):
                        crop = crop_from_source(prev_file, f.get("bbox", [0,0,0,0]), f.get("frame_idx"))
                        if crop:
                            st.image(crop, caption=f"Face {i+1} | score≈{f.get('det_score',0):.2f}", use_container_width=True)
                        else:
                            st.text(f"Face {i+1}: preview not available")
                except Exception as e:
                    st.error(f"Preview failed: {e}")

    with right:
        st.markdown("#### 3) Search by Query Image")
        st.caption("Upload a face image; we’ll detect the largest face and cosine-search the index.")
        query = st.file_uploader("Query face image", type=["jpg","jpeg","png","bmp","webp"], key="face_query_uploader")
        top_k = st.slider("Top-K results", 1, 20, 5, 1)
        if st.button("Search", type="primary", disabled=(query is None)):
            try:
                resp = api_faces_search(query.getvalue(), query.name, int(top_k))
                st.success(f"Faces in query: {resp.get('query_faces_found',0)}")
                results = resp.get("results", [])
                if not results:
                    st.info("No matches returned (index might be empty).")
                else:
                    for i, r in enumerate(results):
                        col_img, col_meta = st.columns([1, 1])
                        with col_img:
                            crop = crop_from_source(r.get("source_file",""), r.get("bbox",[0,0,0,0]), r.get("frame_idx"))
                            if crop:
                                st.image(crop, caption=f"Match {i+1}  |  score={r.get('score',0):.3f}", use_container_width=True)
                            else:
                                st.text(f"Match {i+1}: (preview unavailable)")
                        with col_meta:
                            st.write(f"**Source file:** `{r.get('source_file')}`")
                            st.write(f"**Frame:** {r.get('frame_idx')}")
                            st.write(f"**Timestamp (s):** {r.get('ts_sec')}")
                            st.write(f"**BBox:** {r.get('bbox')}")
            except requests.HTTPError as e:
                try:
                    st.error(f"Search failed: {e.response.text}")
                except Exception:
                    st.error(f"Search failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

st.markdown("---")
st.caption("Next: PDF report export with hashes & snapshots, and basic tamper checks (ELA).")
