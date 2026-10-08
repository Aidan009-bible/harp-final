# 🎵 HarpHand — Harp String Detection

**Nat Shin Naung** · Myanmar Harp String Detection for Teaching & Performance

HarpHand is a web application that detects which strings are plucked during a harp performance using **audio analysis**, **hand tracking**, or **both combined**. Upload a video of a harp session and receive timestamped pluck logs, annotated video overlays, and exportable note sheets.

---

## ✨ Features

- **Audio Detection** — TensorFlow model + YIN pitch estimation to identify plucked strings from audio onsets
- **Hand Detection** — YOLOv8 object detection + MediaPipe hand landmarks to track finger–string contact
- **Combined Mode** — Aligns audio onsets with hand tracking and reports cross-signal agreement for review
- **Annotated Video** — Generates labeled output video with string/finger annotations burned in via FFmpeg
- **CSV & PDF Export** — Download detection logs as CSV, annotated video, or PDF note sheets
- **Optional Google OAuth** — Available when frontend and backend OAuth credentials are configured
- **Real-Time Job Status** — Background processing with live status polling
- **Research Evaluation** — Compares prediction CSVs with labeled events using event, timing, and per-string metrics
- **Validation Calibration** — Selects auditable per-string operating thresholds without retraining the model
- **Run Manifests** — Records model checksum, thresholds, tensor shapes, and runtime versions for reproducibility

---

## 🏗️ Architecture

```
HarpHand/
├── backend/                  # FastAPI server (Python)
│   ├── app.py                # Main API — upload, job management, downloads
│   ├── inference.py          # Audio pipeline (mel spectrogram, model, YIN fallback)
│   ├── calibrate_thresholds.py # Validation-set per-string threshold selection
│   ├── evaluate_model.py     # Ground-truth evaluation CLI
│   ├── core/                 # Configuration, uploads, jobs, metrics, and calibration utilities
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
- **Node.js** 20.19+
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
| CORS origins | `HARP_ALLOWED_ORIGINS` | Optional comma-separated override |
| API development proxy | `VITE_API_PROXY_TARGET` | Backend URL used by Vite; defaults to `http://127.0.0.1:8000` |
| Google OAuth | `VITE_GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Frontend client ID and backend exchange credentials |
| Custom model uploads | `ALLOW_CUSTOM_MODEL_UPLOADS=true` | Disabled by default because model files must be trusted |
| Calibrated thresholds | `HARP_THRESHOLDS_PATH=/path/to/thresholds.json` | Optional validation-selected per-string threshold profile |
| Audio/hand fusion window | `HAND_PRE_ONSET_MS=150` | Visual contacts accepted before an audio onset |
| Upload limits | `MAX_VIDEO_MB`, `MAX_MODEL_MB`, `MAX_WEIGHTS_MB` | Streaming server-side file limits |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload a video and start a detection job; trusted custom model files are optional when explicitly enabled |
| `GET` | `/api/status/{job_id}` | Poll job status (`queued`, `running`, `done`, `error`) |
| `GET` | `/api/download/csv/{job_id}` | Download detection results as CSV |
| `GET` | `/api/download/video/{job_id}` | Download annotated video |
| `GET` | `/api/download/manifest/{job_id}` | Download the audio inference reproducibility manifest |
| `GET` | `/api/video-stream/{job_id}` | Stream annotated video for in-browser preview |
| `GET` | `/api/logs/{job_id}` | Get combined audio + hand event log |
| `GET` | `/api/defaults` | Get model availability and upload capabilities |
| `GET` | `/api/health` | Check backend readiness |
| `POST` | `/api/auth/google` | Google OAuth token verification |

---

## 🎯 Detection Modes

### Audio Only
Extracts audio from the video, computes mel spectrograms, and runs the TensorFlow model to classify which of the 16 harp strings were plucked at each onset. Optionally uses **YIN pitch estimation** as a hybrid fallback.

### Hand Only
Runs YOLOv8 to detect hands in each frame, then uses MediaPipe hand landmarks to identify finger positions and proximity to harp strings.

### Both (Combined)
Runs audio detection first, then hand detection. Hand events are **filtered to pluck moments only** using the configured pre-onset window (150ms by default). Produces a combined video with annotations and subtitles overlaid.

The agreement percentage shown by the interface measures how often audio and hand labels match on comparable events. It is diagnostic evidence, not ground-truth model accuracy. See [the model evaluation protocol](docs/model-evaluation.md) before deciding whether to retrain.

The staged, source-backed improvement plan is documented in [the research roadmap](docs/research-roadmap.md).

---

## ✅ Quality checks

```bash
cd frontend
npm run check

cd ../backend
python -m unittest discover -s tests -v
python -m compileall -q .
```

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
- [Vite 7](https://vite.dev/) — build tool & dev server
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
