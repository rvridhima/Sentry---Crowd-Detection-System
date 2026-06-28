# SENTRY

**AI-Powered Crime Detection in Public Areas**
Crowd Anomaly Detection · Behavioral Analytics · Real-Time Alerts · Video Intelligence

SENTRY is a browser-based surveillance console that analyzes uploaded CCTV footage in real time, detects people using a pretrained YOLOv8 model, flags dangerous crowd clustering and rapid/erratic movement, and surfaces every finding as a live, timestamped alert for a control-room operator.

---

## What it does

1. **Upload footage** — drag-and-drop or browse to upload a CCTV clip (MP4, WebM, MOV).
2. **Real person detection** — YOLOv8 runs on sampled frames and draws a bounding box around every detected person, with a confidence score.
3. **Anomaly detection, driven by real detections** — two independent signals, both anchored to actual person boxes (never to empty regions of frame):
   - **Crowd clustering** — a sliding-window scan flags regions where people are packed densely together.
   - **Rapid movement** — area-weighted motion analysis flags fast, erratic movement (a struggle, a sprint, a sudden scatter) around detected people, independent of cluster size.
4. **Live Threat Assessment** — a 0–100 risk score, fused from density, clustering, and motion, classified into **Normal / Elevated / Critical**, shown on a real-time gauge.
5. **Real-time alert log** — every Elevated or Critical reading is timestamped and logged instantly in the dashboard. The log starts empty and builds up as the session runs; nothing is sent anywhere off-device — alerts are on-screen only.

---

## Project structure

```
.
├── sentry-crowd-detection.html   ← the application — open this in a browser
└── backend/                      ← optional, for higher-accuracy detection
    ├── app.py
    ├── requirements.txt
    └── README.md
```

## Running it

**Quick start — no installation:**
Just open `sentry-crowd-detection.html` in any modern browser (Chrome, Edge, Firefox). On first load it downloads a small pretrained model (YOLOv8n, ~13MB) over the internet, then runs entirely client-side. No server, no build step, no dependencies.

**Optional — higher-accuracy mode:**
A small Python/Flask backend can run a larger YOLOv8 model (medium, by default) for meaningfully better detection accuracy than the in-browser model. This is entirely optional — the app works fully without it. See [`backend/README.md`](backend/README.md) for setup; once running, point the app at it by editing `backendUrl` in the `CONFIG` block near the top of the HTML file's `<script>` section.

---

## How detection works

| Stage | What happens |
|---|---|
| **Frame sampling** | A frame is sampled from the video every 700ms and letterboxed to 640×640 (YOLOv8's required input size) without distorting aspect ratio. |
| **Person detection** | The frame is run through YOLOv8 (in-browser via ONNX Runtime Web, or the backend via PyTorch). Raw output is decoded and passed through a custom Non-Maximum Suppression pass, filtered to the "person" class. |
| **Cluster detection** | A sliding window scans the frame for regions containing several detected people in close proximity — the basis for crowd-anomaly flags. This is purely geometric, computed from real detections, so an empty area can never be flagged. |
| **Motion analysis** | Frame-differencing measures movement intensity per region, weighted by exact geometric overlap with detected people — this is what allows a fight or a sprint to be flagged even without a large crowd present. |
| **Risk fusion** | Density, clustering, and motion signals combine into a single 0–100 risk index. Crossing 45 logs an "Elevated" alert; crossing 72 logs a "Critical" alert. |
| **Held-state display** | The gauge's verdict briefly holds at the highest severity seen in the last few seconds, so it never visually contradicts an alert that was just logged from a momentary spike. |

All of the above is implemented directly in `sentry-crowd-detection.html` — there's no hidden server-side logic in the default configuration.

---

## Technology stack

- **Frontend:** HTML5, CSS3, vanilla JavaScript, Canvas API
- **AI / Computer Vision:** YOLOv8 (Ultralytics, COCO-pretrained), ONNX Runtime Web (WebAssembly)
- **Optional backend:** Python 3, Flask, Flask-CORS, Ultralytics (PyTorch)

---

## Known limitations

- Detection accuracy in the default (no-backend) mode is bounded by YOLOv8**n** ("nano"), the smallest and fastest variant in the YOLOv8 family — chosen so the app needs no installation. The optional backend (or a manually exported larger ONNX model — see the comment on `modelUrl` in the HTML's `CONFIG` block) improves on this.
- This is a detection and alerting aid for a human operator, not an autonomous decision system — all flagged events are meant to be reviewed by a person, not acted on automatically.
- The app requires an internet connection on first load to fetch the detection model; after that, no footage or detection data leaves the browser.

---

## Future Scope

Multi-camera correlation, larger/specialized crowd models, mobile alerts to field units, trajectory and loitering analysis, and formal governance/data-retention safeguards prior to any real-world deployment. See the project presentation for details.
