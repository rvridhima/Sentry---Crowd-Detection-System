"""
SENTRY — Optional high-accuracy detection backend
--------------------------------------------------
Runs a real, full-size Ultralytics YOLOv8 model (PyTorch, not ONNX/WASM) and
exposes it over a small HTTP API that the SENTRY frontend can call instead of
its built-in browser-only detection.

This is OPTIONAL. The single-file HTML console works on its own with no
backend at all (YOLOv8n via ONNX Runtime Web, in-browser). Run this only if
you want a real accuracy upgrade — YOLOv8m/l/x are meaningfully more accurate
than the nano model the browser path uses, at the cost of needing a server
(your laptop is fine; it does not need a GPU, just slower per-frame).

----------------------------------------------------------------------------
SETUP (one time)
----------------------------------------------------------------------------
    pip install -r requirements.txt

The first request will auto-download the chosen model weights (yolov8m.pt by
default, ~50MB) from Ultralytics' servers — this needs internet access once,
then it's cached locally.

----------------------------------------------------------------------------
RUN
----------------------------------------------------------------------------
    python app.py

This starts a server on http://localhost:8000. Leave it running, then in the
SENTRY HTML file's CONFIG block, set:

    backendUrl: "http://localhost:8000/detect"

(See the matching comment in index.html's CONFIG for exactly where this goes.)
If backendUrl is set, the frontend will prefer this backend over the
in-browser model automatically.

----------------------------------------------------------------------------
CHOOSING A MODEL SIZE
----------------------------------------------------------------------------
Set MODEL_NAME below. Larger = more accurate, slower per frame, bigger
download. For a CPU-only laptop, yolov8m is usually the sweet spot; yolov8l
or yolov8x are noticeably slower without a GPU.

    yolov8n  (nano)    fastest, least accurate — same tier as the browser path
    yolov8s  (small)   solid step up, still fast on CPU
    yolov8m  (medium)  meaningfully more accurate, default here
    yolov8l  (large)   better still, slow on CPU
    yolov8x  (xlarge)  most accurate, needs a GPU to be usable in real time
"""

import io
import time

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "yolov8m.pt"   # change to yolov8s.pt / yolov8l.pt / yolov8x.pt as desired
PERSON_CLASS_ID = 0          # COCO class 0 = "person" — same convention as the frontend
CONFIDENCE_THRESHOLD = 0.35  # matches the frontend's default scoreThreshold
IOU_THRESHOLD = 0.45         # matches the frontend's default iouThreshold
PORT = 8000

# ---------------------------------------------------------------------------
# App + model setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # allow the browser-hosted HTML file to call this from a different origin

print(f"Loading {MODEL_NAME} ... (first run downloads weights, may take a moment)")
model = YOLO(MODEL_NAME)
print("Model loaded. Ready for requests.")


@app.route("/health", methods=["GET"])
def health():
    """Simple liveness check — useful to confirm the backend is reachable
    before wiring it into the frontend."""
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/detect", methods=["POST"])
def detect():
    """
    Accepts a single video frame as image bytes (POST body, raw JPEG/PNG,
    content-type doesn't matter much — Pillow sniffs it) and returns person
    detections in the same normalized [0-1] coordinate shape the frontend's
    in-browser YOLO path already produces, so it's a drop-in replacement:

        { "persons": [ { "x": 0.12, "y": 0.30, "w": 0.05, "h": 0.18, "score": 0.91 }, ... ],
          "inferenceMs": 42.3 }
    """
    if not request.data:
        return jsonify({"error": "No image data in request body"}), 400

    try:
        image = Image.open(io.BytesIO(request.data)).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Could not decode image: {exc}"}), 400

    img_w, img_h = image.size

    start = time.time()
    results = model.predict(
        image,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=[PERSON_CLASS_ID],  # ask the model to only return persons
        verbose=False,
    )
    inference_ms = (time.time() - start) * 1000

    persons = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            persons.append({
                "x": max(0.0, x1 / img_w),
                "y": max(0.0, y1 / img_h),
                "w": min(1.0, (x2 - x1) / img_w),
                "h": min(1.0, (y2 - y1) / img_h),
                "score": round(score, 4),
            })

    return jsonify({"persons": persons, "inferenceMs": round(inference_ms, 1)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
