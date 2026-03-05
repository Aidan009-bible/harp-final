# 🎵 HarpHand — Harp String Detection

**Nat Shin Naung** · Myanmar Harp String Detection for Teaching & Performance

HarpHand is a web application that detects which strings are plucked during a harp performance using **audio analysis**, **hand tracking**, or **both combined**. Upload a video of a harp session and receive timestamped pluck logs, annotated video overlays, and exportable note sheets.

---

## ✨ Features

- **Audio Detection** — TensorFlow model + YIN pitch estimation to identify plucked strings from audio onsets
- **Hand Detection** — YOLOv8 object detection + MediaPipe hand landmarks to track finger–string contact
- **Combined Mode** — Merges audio onsets with hand tracking; filters hand events to pluck moments for high-accuracy results
- **Annotated Video** — Generates labeled output video with string/finger annotations burned in via FFmpeg
- **CSV & PDF Export** — Download detection logs as CSV, annotated video, or PDF note sheets
- **Google OAuth Login** — Sign in with Google or email/password
- **Real-Time Job Status** — Background processing with live status polling

---

## 🏗️ Architecture

```
HarpHand/
├── backend/                  # FastAPI server (Python)
│   ├── app.py                # Main API — upload, job management, downloads
│   ├── inference.py          # Audio pipeline (mel spectrogram, model, YIN fallback)
│   ├── harp_hand_detector.py # Hand/finger detection (YOLOv8 + MediaPipe)
│   ├── hand_landmarker.task  # MediaPipe hand landmark model
│   ├── models/               # .keras model files (audio)
│   ├── weights/              # .pt weight files (YOLO hand detection)
│   └── requirements.txt
├── frontend/                 # React + Vite (JavaScript)
│   ├── src/
│   │   ├── App.jsx           # Router (Home, Login, Tool)
│   │   ├── pages/
│   │   │   ├── Home.jsx      # Landing page
│   │   │   ├── Login.jsx     # Authentication (Google OAuth + email)
│   │   │   └── Tool.jsx      # Main detection tool UI
│   │   └── index.css         # Global styles
│   ├── vite.config.js
│   └── package.json
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10+
- **Node.js** 18+
- **FFmpeg** (installed and on PATH)

### Backend Setup

```bash
cd HarpHand/backend

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Place model files (if not using upload):
#   backend/models/default.keras  — audio detection model
#   backend/weights/best.pt       — YOLOv8 hand detection weights

# Start the server
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### Frontend Setup

```bash
cd HarpHand/frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

The frontend runs at **http://localhost:5173** and proxies API requests to the backend.

---

## 🔧 Configuration

| Item | Location | Description |
|------|----------|-------------|
| Audio model | `backend/models/default.keras` | TensorFlow/Keras model for string classification |
| Hand weights | `backend/weights/best.pt` | YOLOv8 weights for hand/finger detection |
| CORS origins | `backend/app.py` | Allowed frontend origins |
| API proxy | `frontend/vite.config.js` | Backend URL for dev proxy |
| Google OAuth | `frontend/src/pages/Login.jsx` | Google OAuth client ID |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload video + model/weights, start detection job |
| `GET` | `/api/status/{job_id}` | Poll job status (`queued`, `running`, `done`, `error`) |
| `GET` | `/api/download/csv/{job_id}` | Download detection results as CSV |
| `GET` | `/api/download/video/{job_id}` | Download annotated video |
| `GET` | `/api/video-stream/{job_id}` | Stream annotated video for in-browser preview |
| `GET` | `/api/logs/{job_id}` | Get combined audio + hand event log |
| `POST` | `/api/auth/google` | Google OAuth token verification |

---

## 🎯 Detection Modes

### Audio Only
Extracts audio from the video, computes mel spectrograms, and runs the TensorFlow model to classify which of the 16 harp strings were plucked at each onset. Optionally uses **YIN pitch estimation** as a hybrid fallback.

### Hand Only
Runs YOLOv8 to detect hands in each frame, then uses MediaPipe hand landmarks to identify finger positions and proximity to harp strings.

### Both (Combined)
Runs audio detection first, then hand detection. Hand events are **filtered to pluck moments only** — only hand/finger annotations within 150ms before each audio onset are retained. Produces a combined video with annotations and subtitles overlaid.

---

## 🛠️ Tech Stack

**Backend:**
- [FastAPI](https://fastapi.tiangolo.com/) — async Python web framework
- [TensorFlow/Keras](https://www.tensorflow.org/) — audio classification model
- [Librosa](https://librosa.org/) — audio processing & onset detection
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) — hand object detection
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) — hand landmark tracking
- [OpenCV](https://opencv.org/) — video frame processing & annotation
- [FFmpeg](https://ffmpeg.org/) — video encoding & subtitle overlay

**Frontend:**
- [React 18](https://react.dev/) — UI framework
- [Vite 5](https://vitejs.dev/) — build tool & dev server
- [React Router](https://reactrouter.com/) — client-side routing
- [@react-oauth/google](https://www.npmjs.com/package/@react-oauth/google) — Google sign-in
- [jsPDF](https://github.com/parallax/jsPDF) + [html2canvas](https://html2canvas.hertzen.com/) — PDF export

**Deployment:**
- **Frontend** → [Vercel](https://vercel.com/)
- **Backend** → AWS EC2

---

## 📝 License

This project is developed for educational and research purposes.

---

<p align="center">
  <strong>Nat Shin Naung</strong> · Myanmar Harp String Detection
</p>
