# SENTRY — Optional High-Accuracy Detection Backend

The main `sentry-crowd-detection.html` file works
completely on its own with no setup — it runs YOLOv8n (the smallest YOLOv8
model - nano) entirely in-browser. The backend can be used for an
accuracy upgrade which runs a small local server.

## Why this exists

YOLOv8 comes in five sizes — nano, small, medium, large, x-large — and each
step up is a real, measurable accuracy improvement (nano trades roughly 7-8
points of detection accuracy for being the smallest and fastest). The
browser-only version is stuck with nano because that's the only size with a
freely hosted, browser-loadable `.onnx` file available right now. This
backend runs the **medium** model (configurable) using the real
`ultralytics` PyTorch library — no browser/WASM size constraints — and
exposes it over a tiny local HTTP API.

## Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```

First run downloads the model weights (~50MB for medium) automatically —
needs internet once, then it's cached. Leave this running; it serves on
`http://localhost:8000`.

## Connect it to the frontend

Open `sentry-crowd-detection.html`, find the `CONFIG` block near the top of
the `<script>` section, and set:

```js
backendUrl: "http://localhost:8000/detect",
```

Reload the page. The console will check the backend on startup and use it
automatically — you'll see "Backend ready — yolov8m.pt active" in the model
status badge instead of the usual YOLOv8n badge. If the backend isn't
running or unreachable, the page falls back to the in-browser model
automatically — nothing breaks if you forget to start it.

## Choosing a model size

Edit `MODEL_NAME` near the top of `app.py`:

| Model | Relative accuracy | Speed on CPU |
|---|---|---|
| `yolov8n.pt` | baseline (same as browser path) | fastest |
| `yolov8s.pt` | solid step up | fast |
| `yolov8m.pt` | meaningfully more accurate (**default here**) | moderate |
| `yolov8l.pt` | better still | slow without a GPU |
| `yolov8x.pt` | most accurate | needs a GPU for real-time use |
