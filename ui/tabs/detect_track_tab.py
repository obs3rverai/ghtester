# ui/tabs/detect_track_tab.py
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


import os

def _get_api_base() -> str:
    # 1) Streamlit secrets (if present)
    try:
        return st.secrets["API_BASE"]  # will raise if secrets file missing
    except Exception:
        pass
    # 2) Environment variable (optional)
    envv = os.environ.get("API_BASE")
    if envv:
        return envv
    # 3) Default to local FastAPI
    return "http://127.0.0.1:8000/api"

API_BASE = _get_api_base()


# Where the backend writes files (so Streamlit can open them directly)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ANNOT_DIR = DATA_DIR / "annotated"
DETS_DIR  = DATA_DIR / "detections"
TRACKS_DIR = DATA_DIR / "tracks"

# Session keys used by the Report tab
FINDINGS_KEY = "report_findings"


def _init_state():
    if FINDINGS_KEY not in st.session_state:
        st.session_state[FINDINGS_KEY] = []  # list of findings dicts for the report


def _media_block(title: str, annotated_path: Optional[str]):
    st.subheader(title)
    if not annotated_path:
        st.info("No annotated media generated.")
        return
    p = Path(annotated_path)
    if not p.exists():
        st.warning(f"File not found on disk: {p}")
        return
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        st.image(str(p), caption=p.name, use_container_width=True)
    elif p.suffix.lower() in {".mp4", ".mov", ".m4v", ".webm"}:
        st.video(str(p))
    else:
        st.write(f"Saved: `{p}`")


def _json_viewer(payload: Dict[str, Any], label: str, height: int = 220):
    st.markdown(f"**{label}**")
    st.json(payload, expanded=False)


def _api_post(url: str, files: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, files=files, data=data, timeout=600)
    if not r.ok:
        raise RuntimeError(f"{url} failed ({r.status_code}): {r.text}")
    return r.json()


def _add_findings(findings: List[Dict[str, Any]]) -> int:
    """Append findings to the report session list, return count added."""
    if not findings:
        return 0
    st.session_state[FINDINGS_KEY].extend(findings)
    return len(findings)


def render():
    _init_state()

    st.header("Detection & Tracking")
    st.caption("Run YOLO for people & vehicles; preview annotated results; add Findings to your report.")
    tabs = st.tabs(["Detect (Image)", "Detect (Video)", "Track (Video)"])

    # ---------------- Detect (Image) ----------------
    with tabs[0]:
        st.subheader("Detect objects in an image")
        img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="det_img_up")
        colA, colB, colC = st.columns(3)
        conf = colA.slider("Confidence", 0.1, 0.9, 0.35, 0.01, help="YOLO score threshold")
        iou  = colB.slider("NMS IoU", 0.1, 0.9, 0.50, 0.01, help="Non-maximum suppression IoU")
        model = colC.selectbox("Model", ["yolov8l.pt", "yolov8x.pt"], index=0, key="det_img_model")

        if st.button("Run detection on image", use_container_width=True, key="det_img_btn"):
            if img_file is None:
                st.error("Please upload an image first.")
            else:
                try:
                    res = _api_post(
                        f"{API_BASE}/detect/image",
                        files={"image": (img_file.name, img_file.getvalue(), img_file.type)},
                        data={"model_name": model, "conf": conf, "iou": iou},
                    )
                    _json_viewer(res, "Detection result")
                    _media_block("Annotated image", res.get("annotated_path"))
                    st.success(f"Detections: {res.get('total_detections', 0)}")
                    st.info("Tip: Tracking (next tabs) will produce Findings with track IDs for the report.")
                except Exception as e:
                    st.error(f"Detection failed: {e}")

    # ---------------- Detect (Video) ----------------
    with tabs[1]:
        st.subheader("Detect objects in a video")
        vid_file = st.file_uploader("Upload video", type=["mp4", "mov", "m4v", "webm"], key="det_vid_up")
        colA, colB, colC, colD = st.columns(4)
        conf = colA.slider("Confidence", 0.1, 0.9, 0.35, 0.01, key="det_vid_conf")
        iou  = colB.slider("NMS IoU", 0.1, 0.9, 0.50, 0.01, key="det_vid_iou")
        stride = colC.number_input("Frame stride", min_value=1, max_value=8, value=1, step=1, help="Process every Nth frame")
        maxf   = colD.number_input("Max frames", min_value=0, max_value=5000, value=0, step=50, help="0 = no cap")
        model = st.selectbox("Model", ["yolov8l.pt", "yolov8x.pt"], index=0, key="det_vid_model")

        if st.button("Run detection on video", use_container_width=True, key="det_vid_btn"):
            if vid_file is None:
                st.error("Please upload a video first.")
            else:
                try:
                    data = {
                        "model_name": model, "conf": conf, "iou": iou,
                        "stride": int(stride),
                        "max_frames": None if int(maxf) == 0 else int(maxf),
                    }
                    files = {"video": (vid_file.name, vid_file.getvalue(), vid_file.type)}
                    res = _api_post(f"{API_BASE}/detect/video", files=files, data=data)
                    _json_viewer(res, "Detection result")
                    _media_block("Annotated video", res.get("annotated_path"))
                    st.success(f"Frames: {res.get('num_frames', 0)} — Detections: {res.get('total_detections', 0)}")
                    st.info("Tip: Use Tracking (next tab) to create Findings with track IDs for the report.")
                except Exception as e:
                    st.error(f"Video detection failed: {e}")

    # ---------------- Track (Video) ----------------
    with tabs[2]:
        st.subheader("Track people & vehicles in a video (ByteTrack)")
        vid_file = st.file_uploader("Upload video", type=["mp4", "mov", "m4v", "webm"], key="trk_vid_up")
        colA, colB, colC = st.columns(3)
        conf = colA.slider("Confidence", 0.1, 0.9, 0.35, 0.01, key="trk_conf")
        iou  = colB.slider("NMS IoU", 0.1, 0.9, 0.50, 0.01, key="trk_iou")
        minlen = colC.number_input("Min track length (frames)", min_value=1, max_value=100, value=5, step=1, help="Shorter tracks are dropped")
        model = st.selectbox("Model", ["yolov8l.pt", "yolov8x.pt"], index=0, key="trk_model")
        device = st.selectbox("Device", ["cpu"], index=0, key="trk_device", help="Codespaces CPU; set '0' if you add a GPU later")

        run = st.button("Run tracking on video", use_container_width=True, key="trk_run")
        if run:
            if vid_file is None:
                st.error("Please upload a video first.")
            else:
                try:
                    data = {
                        "model_name": model, "conf": conf, "iou": iou,
                        "min_track_len": int(minlen),
                        "device": None if device == "cpu" else device,
                    }
                    files = {"video": (vid_file.name, vid_file.getvalue(), vid_file.type)}
                    res = _api_post(f"{API_BASE}/track/video", files=files, data=data)

                    _json_viewer(res, "Tracking result")
                    _media_block("Annotated tracking video", res.get("annotated_path"))

                    # Load the derived findings JSON and offer to add to report
                    findings_json_path = res.get("findings_json_path")
                    findings = []
                    if findings_json_path and Path(findings_json_path).exists():
                        try:
                            with open(findings_json_path, "r", encoding="utf-8") as f:
                                payload = json.load(f)
                            findings = payload.get("findings", [])
                            st.success(f"Tracks: {res.get('total_tracks', 0)} — Found {len(findings)} event(s)")
                            with st.expander("Preview derived Findings"):
                                st.json(findings, expanded=False)
                        except Exception as e:
                            st.warning(f"Could not read findings: {e}")

                    if findings:
                        if st.button("Add these Findings to the Report", key="trk_add_findings"):
                            n = _add_findings(findings)
                            st.toast(f"Added {n} finding(s) to report session.", icon="✅")
                            st.info("Go to the Report tab to generate the PDF. The Findings table will include these events.")

                except Exception as e:
                    st.error(f"Tracking failed: {e}")
