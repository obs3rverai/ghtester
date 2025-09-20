# ui/tabs/detect_track_tab.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

def _get_api_base() -> str:
    try:
        return st.secrets["API_BASE"]
    except Exception:
        pass
    envv = os.environ.get("API_BASE")
    if envv: return envv
    return "http://127.0.0.1:8000/api"

API_BASE = _get_api_base()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# -------- Helpers --------
def _api_get(path: str) -> Dict[str, Any]:
    r = requests.get(f"{API_BASE}{path}", timeout=60)
    r.raise_for_status()
    return r.json()

def _api_post(url: str, files: Dict[str, Any] | None, data: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, files=files, data=data, timeout=1200)
    if not r.ok:
        raise RuntimeError(f"{url} failed ({r.status_code}): {r.text}")
    return r.json()

def _normalize_path(p: Optional[str]) -> Optional[str]:
    if not p: return None
    try:
        ab = (PROJECT_ROOT / p).resolve() if not p.startswith("/") else Path(p).resolve()
        # prefer project-relative for Streamlit display
        try:
            return str(ab.relative_to(PROJECT_ROOT))
        except Exception:
            return str(ab)
    except Exception:
        return p

def _media_block(title: str, annotated_path: Optional[str]):
    st.subheader(title)
    if not annotated_path:
        st.info("No annotated media generated.")
        return

    # 1) Normalize to absolute path under the repo root
    rel = _normalize_path(annotated_path) or annotated_path
    p = (PROJECT_ROOT / rel).resolve() if not str(rel).startswith("/") else Path(rel).resolve()

    # 2) Quick diagnostics (helps during demos)
    exists = p.exists()
    st.caption(f"Path: `{p}`  •  exists: **{exists}**")
    if not exists:
        st.warning("Annotated file not found on disk. Check the backend write location and path returned.")
        return

    # 3) Render: image/video/other
    suf = p.suffix.lower()
    try:
        if suf in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            st.image(str(p), caption=p.name, use_container_width=True)
            return
        if suf in {".mp4", ".mov", ".m4v", ".webm"}:
            # First try direct file path…
            try:
                st.video(str(p))
            except Exception:
                # …fallback: read bytes (more robust in some sandboxes)
                with p.open("rb") as f:
                    st.video(f.read())
            return

        # Unknown type: just show a link
        st.write(f"Saved: `{p}`")
    except Exception as e:
        st.error(f"Could not preview media: {e}")

def _json_viewer(payload: Dict[str, Any], label: str):
    st.markdown(f"**{label}**")
    st.json(payload, expanded=False)

def _list_stored_files() -> List[str]:
    # reuse your existing /forensics/files index from the dashboard code
    try:
        res = _api_get("/forensics/files")
        return res.get("files", [])
    except Exception:
        return []

# Keep Findings in session for the Report tab
FINDINGS_KEY = "report_findings"
def _init_state():
    if FINDINGS_KEY not in st.session_state:
        st.session_state[FINDINGS_KEY] = []

def _add_findings(findings: List[Dict[str, Any]]) -> int:
    if not findings: return 0
    st.session_state[FINDINGS_KEY].extend(findings)
    return len(findings)

# -------- UI --------
def render():
    _init_state()
    st.header("Detection & Tracking")
    st.caption("Run YOLO for people & vehicles; preview annotated results; add Findings to your report.")

    tabs = st.tabs(["Detect (Image)", "Detect (Video)", "Track (Video)"])

    # ===== Detect (Image) =====
    with tabs[0]:
        st.subheader("Detect objects in an image")
        src = st.radio("Source", ["Upload", "Stored"], horizontal=True, key="det_img_source")

        colA, colB, colC = st.columns(3)
        conf = colA.slider("Confidence", 0.1, 0.9, 0.35, 0.01)
        iou  = colB.slider("NMS IoU", 0.1, 0.9, 0.50, 0.01)
        model = colC.selectbox("Model", ["yolov8l.pt", "yolov8x.pt"], index=0)

        if src == "Upload":
            img_file = st.file_uploader("Upload image", type=["jpg","jpeg","png","bmp","webp"], key="det_img_up")
            if st.button("Run detection on image (upload)", use_container_width=True):
                if not img_file:
                    st.error("Please upload an image.")
                else:
                    res = _api_post(
                        f"{API_BASE}/detect/image",
                        files={"image": (img_file.name, img_file.getvalue(), img_file.type)},
                        data={"model_name": model, "conf": conf, "iou": iou},
                    )
                    _json_viewer(res, "Detection result")
                    _media_block("Annotated image", res.get("annotated_path"))
        else:
            stored = _list_stored_files()
            imgs = [f for f in stored if Path(f).suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}]
            pick = st.selectbox("Stored image", ["-- select --"] + imgs, index=0, key="det_img_pick")
            if st.button("Run detection on stored image", use_container_width=True, disabled=(pick=="-- select --")):
                res = _api_post(
                    f"{API_BASE}/detect/image/by-filename/{pick}",
                    files=None,
                    data={"model_name": model, "conf": conf, "iou": iou},
                )
                _json_viewer(res, "Detection result")
                _media_block("Annotated image", res.get("annotated_path"))

    # ===== Detect (Video) =====
    with tabs[1]:
        st.subheader("Detect objects in a video")
        src = st.radio("Source", ["Upload", "Stored"], horizontal=True, key="det_vid_source")

        colA, colB, colC, colD = st.columns(4)
        conf = colA.slider("Confidence", 0.1, 0.9, 0.35, 0.01, key="det_vid_conf")
        iou  = colB.slider("NMS IoU", 0.1, 0.9, 0.50, 0.01, key="det_vid_iou")
        stride = colC.number_input("Frame stride", min_value=1, max_value=8, value=1, step=1)
        maxf   = colD.number_input("Max frames", min_value=0, max_value=5000, value=0, step=50, help="0 = no cap")
        model = st.selectbox("Model", ["yolov8l.pt", "yolov8x.pt"], index=0, key="det_vid_model")

        if src == "Upload":
            vid_file = st.file_uploader("Upload video", type=["mp4","mov","m4v","webm"], key="det_vid_up")
            if st.button("Run detection on video (upload)", use_container_width=True, key="det_vid_btn_up"):
                if not vid_file:
                    st.error("Please upload a video.")
                else:
                    data = {
                        "model_name": model, "conf": conf, "iou": iou,
                        "stride": int(stride),
                        "max_frames": None if int(maxf) == 0 else int(maxf),
                    }
                    res = _api_post(f"{API_BASE}/detect/video", files={"video": (vid_file.name, vid_file.getvalue(), vid_file.type)}, data=data)
                    _json_viewer(res, "Detection result")
                    _media_block("Annotated video", res.get("annotated_path"))
        else:
            stored = _list_stored_files()
            vids = [f for f in stored if Path(f).suffix.lower() in {".mp4",".mov",".m4v",".webm",".mkv",".avi"}]
            pick = st.selectbox("Stored video", ["-- select --"] + vids, index=0, key="det_vid_pick")
            if st.button("Run detection on stored video", use_container_width=True, key="det_vid_btn_stored", disabled=(pick=="-- select --")):
                data = {
                    "model_name": model, "conf": conf, "iou": iou,
                    "stride": int(stride),
                    "max_frames": None if int(maxf) == 0 else int(maxf),
                }
                res = _api_post(f"{API_BASE}/detect/video/by-filename/{pick}", files=None, data=data)
                _json_viewer(res, "Detection result")
                _media_block("Annotated video", res.get("annotated_path"))

    # ===== Track (Video) =====
    with tabs[2]:
        st.subheader("Track people & vehicles in a video (ByteTrack)")
        src = st.radio("Source", ["Upload", "Stored"], horizontal=True, key="trk_vid_source")

        colA, colB, colC = st.columns(3)
        conf = colA.slider("Confidence", 0.1, 0.9, 0.35, 0.01, key="trk_conf")
        iou  = colB.slider("NMS IoU", 0.1, 0.9, 0.50, 0.01, key="trk_iou")
        minlen = colC.number_input("Min track length (frames)", min_value=1, max_value=100, value=5, step=1)
        model = st.selectbox("Model", ["yolov8l.pt", "yolov8x.pt"], index=0, key="trk_model")
        device = st.selectbox("Device", ["cpu"], index=0, key="trk_device")

        if src == "Upload":
            vid_file = st.file_uploader("Upload video", type=["mp4","mov","m4v","webm"], key="trk_vid_up")
            if st.button("Run tracking on video (upload)", use_container_width=True, key="trk_run_up"):
                if not vid_file:
                    st.error("Please upload a video.")
                else:
                    data = {
                        "model_name": model, "conf": conf, "iou": iou,
                        "min_track_len": int(minlen),
                        "device": None if device == "cpu" else device,
                    }
                    res = _api_post(f"{API_BASE}/track/video", files={"video": (vid_file.name, vid_file.getvalue(), vid_file.type)}, data=data)
                    _json_viewer(res, "Tracking result")
                    _media_block("Annotated tracking video", res.get("annotated_path"))
                    _maybe_add_findings(res)
        else:
            stored = _list_stored_files()
            vids = [f for f in stored if Path(f).suffix.lower() in {".mp4",".mov",".m4v",".webm",".mkv",".avi"}]
            pick = st.selectbox("Stored video", ["-- select --"] + vids, index=0, key="trk_vid_pick")
            if st.button("Run tracking on stored video", use_container_width=True, key="trk_run_stored", disabled=(pick=="-- select --")):
                data = {
                    "model_name": model, "conf": conf, "iou": iou,
                    "min_track_len": int(minlen),
                    "device": None if device == "cpu" else device,
                }
                res = _api_post(f"{API_BASE}/track/video/by-filename/{pick}", files=None, data=data)
                _json_viewer(res, "Tracking result")
                _media_block("Annotated tracking video", res.get("annotated_path"))
                _maybe_add_findings(res)

def _maybe_add_findings(res: Dict[str, Any]):
    findings_json_path = res.get("findings_json_path")
    findings = []
    try:
        if findings_json_path:
            p = Path(_normalize_path(findings_json_path) or "")
            if not p.is_absolute():
                p = (PROJECT_ROOT / p).resolve()
            if p.exists():
                payload = json.loads(Path(p).read_text(encoding="utf-8"))
                findings = payload.get("findings", [])
    except Exception as e:
        st.warning(f"Could not read findings: {e}")

    if findings:
        with st.expander("Preview derived Findings"):
            st.json(findings, expanded=False)
        if st.button("Add these Findings to the Report", key=f"trk_add_findings_{hash(findings_json_path)}"):
            n = _add_findings(findings)
            st.toast(f"Added {n} finding(s) to report session.", icon="✅")
            st.info("Open the Report tab to generate the PDF with these events.")
