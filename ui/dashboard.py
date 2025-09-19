# ui/dashboard.py
import json
import mimetypes
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

# --- config ---
API_BASE = "http://127.0.0.1:8000"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="AI CCTV & Digital Media Forensic Tool — MVP", layout="wide")
st.title("AI CCTV & Digital Media Forensic Tool — MVP")
st.caption("Hackathon demo UI (uploads, hash + file list).")

# --- helpers ---
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

def try_show_media(stored_path: str):
    """
    Try to preview the uploaded media from the stored local path.
    Works if Streamlit and the API share the same filesystem (local dev).
    """
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


# --- layout ---
health_col, files_col = st.columns([1, 2])

with health_col:
    st.subheader("Server Health")
    health = api_health()
    if health.get("status") == "ok":
        st.success("Backend is running ✅")
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
st.subheader("Upload & Hash")

uploaded = st.file_uploader("Select an image/video to ingest", type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "mov", "mkv", "avi"])
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

st.markdown("---")
st.caption("Next steps: detection/tracking, face embeddings, FAISS search, forensics & reports.")
