📹 CCTV Monitoring & Dashboard

This repository is a prototype for CCTV analytics and visualization, developed as part of the Goa Police Hackathon 2025.
It combines a FastAPI backend with a Streamlit dashboard for monitoring video analytics, running YOLO object detection models, and displaying results interactively.

🚀 Features

FastAPI backend (app/) serving detection and analytics APIs.

Streamlit dashboard (ui/) for live monitoring, visualization, and control.

YOLOv8 models (yolov8l.pt, yolov8n.pt) integrated for object detection.

Modular design separating backend, frontend, and data handling.

📂 Project Structure
.
├── app/                # FastAPI backend
│   └── main.py
├── ui/                 # Streamlit dashboard
│   └── dashboard.py
├── data/               # Sample / input data
├── yolov8l.pt          # YOLOv8 large model (⚠️ large file, use Git LFS / external hosting)
├── yolov8n.pt          # YOLOv8 nano model
├── requirements.txt    # Python dependencies
└── README.md           # This file

🛠️ Setup
1. Clone the repository
git clone https://github.com/adwaitdeshpande-and/ghtest.git
cd ghtest

2. Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate     # Linux / macOS
# .venv\Scripts\activate      # Windows

3. Install dependencies
pip install -r requirements.txt

▶️ Running the project

Open two terminals in the project root:

Terminal 1 → Start FastAPI backend
uvicorn app.main:app --reload


Backend runs at: http://127.0.0.1:8000

Terminal 2 → Start Streamlit dashboard
streamlit run ui/dashboard.py


Dashboard runs at: http://localhost:8501