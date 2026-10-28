# Motiongate

> **High-Throughput Multi-Camera Edge Vision & Face Recognition Pipeline**  
> Written in **Go** and **Python** with zero-cost motion gating, real-time tracking, zero-trust best-shot selection, and ArcFace biometric recognition.

---

## Overview & Architecture

Running continuous deep learning inference (YOLO, InsightFace, ArcFace) across multiple 1080p/4K RTSP security streams at 15–30 FPS quickly overwhelms GPUs and CPUs. Furthermore, in access control and security monitoring, sending 100 blurry, side-profile frames of someone walking past a camera creates redundant computations and increases false biometric matches.

**Motiongate** solves this with an intelligent, multi-stage **processing funnel**:

```
[ RTSP Cameras / Video Files ]
              │
              ▼ (ffmpeg decode pipe)
    ┌──────────────────────────┐
    │ Tier 0: Motion Gate (Go) │ ────── No Motion ──► [ DROP ] (Saves 80–95% CPU)
    └─────────────┬────────────┘
                  │ Motion detected
                  ▼
    ┌──────────────────────────┐
    │ Tier 1: MediaPipe (Py)   │ ◄─── (EfficientDet + BlazeFace on HTTP :8765)
    └─────────────┬────────────┘      • Forehead & YCrCb skin-tone verification
                  │ Person + Face found
                  ▼
    ┌──────────────────────────┐
    │ Tier 2: Tracker (Go)     │ ◄─── Assigns consistent IDs (#1, #2) across frames
    └─────────────┬────────────┘
                  │
                  ▼
    ┌──────────────────────────┐
    │ Tier 3: Best-Shot Engine │ ◄─── Multi-factor quality scoring (Frontality,
    └─────────────┬────────────┘      Laplacian sharpness, Luma, Ceiling-angle penalty)
                  │ Single optimal crop per person
                  ▼
    ┌──────────────────────────┐
    │ Tier 4: Queue & Worker   │ ───► In-Memory Ring Buffer OR RabbitMQ AMQP
    └─────────────┬────────────┘
                  │
                  ▼
    ┌──────────────────────────┐
    │ Tier 5: Face Recog (Py)  │ ◄─── InsightFace ArcFace (512-D embeddings)
    └─────────────┬────────────┘      • SQLite matching (`data/faces.db`) with margin check
                  │
                  ▼
    ┌──────────────────────────┐
    │ Tier 6: Live Web UI      │ ◄─── Real-time SSE dashboard at http://localhost:8080
    └──────────────────────────┘
```

1. **Tier 0 (Fast Motion Gating in pure Go):** Downsamples frames and performs 3-frame temporal differencing ($k+1$ vs $k-1$). Still scenes are dropped immediately before any neural network runs.
2. **Tier 1 (Person & Face Detection via MediaPipe):** Triggers only on moving frames. Employs anatomical and skin-luma filters to reject false positives (e.g. back-of-head hair pareidolia).
3. **Tier 2 (Centroid / IoU Tracking in Go):** Tracks moving individuals across frames using IoU overlap and center-distance fallbacks.
4. **Tier 3 (Zero-Trust Best-Shot Engine):** Accumulates candidates across a person's trajectory and extracts the single crispest, most frontal face/upper-body crop when they leave view (or periodically for long-duration tracks).
5. **Tier 4 (Decoupled Queue):** Handled via an in-memory queue or RabbitMQ for distributed downstream consumers.
6. **Tier 5 (InsightFace Recognition):** Computes 512-D ArcFace vectors and queries enrolled identities in SQLite with configurable similarity thresholds and separation margins.
7. **Tier 6 (SSE Live Monitoring):** Real-time web UI showing camera metrics, bounding boxes, pipeline drop ratios, and a live face recognition audit gallery.

---

## Requirements & Prerequisites

- **Go:** `1.22+` (no external Go dependencies required for the default build!)
- **Python:** `3.9` – `3.12`
- **ffmpeg:** Must be installed and available in system `PATH` (used for RTSP/video decoding)
- **Optional:** Docker (if using RabbitMQ for queueing)

Check prerequisites:
```sh
go version
python3 --version
ffmpeg -version
```

---

## Quick Start Guide

### Step 1: Clone the Repository

```sh
git clone https://github.com/muhammadtalha-cmyk/motion-detect-pipe-line-version-1.git
cd motion-detect-pipe-line-version-1
```

---

### Step 2: Run Motion Gate Only (Pure Go)

You can test the core motion-gating engine with zero Python dependencies using generated synthetic test video:

```sh
# 1. Generate synthetic test video (20s video with moving red square)
./make_test_video.sh

# 2. Run motiongate
go run . -src test.mp4 -loop
```
Open **`http://localhost:8080`** in your browser.  
- The **Left Panel** displays the live camera stream with motion heatmaps.
- The **Right Panel** shows only frames passed through the gate (freezing when the scene is still).
- Notice the **Saved for YOLO** percentage displaying dropped vs. queued frames.

To run against an RTSP camera stream:
```sh
go run . -src "rtsp://username:password@192.168.1.50:554/stream1"
```

---

### Step 3: Run with Person & Face Detection (MediaPipe)

To enable **Person Mode** (filtering out pets, cars, moving trees, and detecting faces):

1. **Set up the MediaPipe Python environment:**
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start the MediaPipe service (Terminal 1):**
   ```sh
   source .venv/bin/activate
   python3 mediapipe_service.py --port 8765
   ```
   *(Models `efficientdet_lite0.tflite` and `blaze_face_short_range.tflite` are included in the repo)*.

3. **Start Motiongate with MediaPipe (Terminal 2):**
   ```sh
   go run . -src test.mp4 -loop -mp http://localhost:8765 -face
   ```

---

### Step 4: Full Pipeline with Face Recognition (InsightFace)

For identity matching against enrolled faces:

1. **Set up the Recognition Python environment:**
   *(Use a separate virtual environment to avoid dependency clashes between MediaPipe and ONNX/InsightFace)*:
   ```sh
   python3 -m venv .venv-rec
   source .venv-rec/bin/activate
   pip install -r requirements-rec.txt
   ```

2. **Start the Recognition Service (Terminal 1):**
   ```sh
   source .venv-rec/bin/activate
   python3 recognizer_service.py --port 8766 --threshold 0.35 --margin 0.05
   ```
   *(On first run, InsightFace automatically downloads the `buffalo_l` model pack ~280 MB into `~/.insightface/models`)*.

3. **Enroll People:**
   Place photos in `enroll_photos/` (e.g. `enroll_photos/Alice.jpg` or `enroll_photos/Alice/photo1.jpg`):
   ```sh
   python3 enroll.py --dir enroll_photos
   # Verify enrolled identities:
   python3 enroll.py --list
   ```

4. **Start the MediaPipe Service (Terminal 2):**
   ```sh
   source .venv/bin/activate
   python3 mediapipe_service.py --port 8765
   ```

5. **Start Motiongate (Terminal 3):**
   ```sh
   go run . -src /path/to/video.mp4 -loop \
            -mp http://localhost:8765 \
            -face \
            -rec http://localhost:8766
   ```

Now open **`http://localhost:8080`**. As people walk in front of the camera:
- Their track trajectory is monitored.
- When they leave view (or every 10 seconds), the **single best shot** is cropped.
- InsightFace identifies the person, and their photo, name, and similarity score appear in the live web gallery!

---

## Multi-Camera Setup

You can run dozens of cameras in parallel through a single unified process. Create or edit `cameras.json` (see [cameras.example.json](cameras.example.json)):

```json
{
  "defaults": {
    "fps": 15,
    "width": 640,
    "tau": 25,
    "theta": 0.01,
    "hold": 2.0,
    "loop": true
  },
  "cameras": [
    { "id": "entry_gate", "src": "rtsp://admin:pass@192.168.1.101:554/h264" },
    { "id": "hallway",    "src": "rtsp://admin:pass@192.168.1.102:554/h264", "theta": 0.008 },
    { "id": "backdoor",   "src": "test.mp4", "hold": 3.0 }
  ]
}
```

Run with configuration:
```sh
go run . -config cameras.json -mp http://localhost:8765 -face -rec http://localhost:8766
```

---

## Distributed Queueing with RabbitMQ

To decouple camera capture from downstream GPU workers:

1. **Start RabbitMQ:**
   ```sh
   docker run -d -p 5672:5672 -p 15672:15672 rabbitmq:3-management
   ```

2. **Run Motiongate with RabbitMQ enabled:**
   ```sh
   go get github.com/rabbitmq/amqp091-go
   go run -tags rabbitmq . -src test.mp4 -loop -queue rabbit -qname frames
   ```

3. **Sample Python Consumer:**
   ```python
   import pika, numpy as np, cv2

   conn = pika.BlockingConnection(pika.URLParameters("amqp://guest:guest@localhost:5672/"))
   ch = conn.channel()
   ch.queue_declare("frames", durable=True)
   ch.basic_qos(prefetch_count=1)

   def callback(ch, method, properties, body):
       cam_id = properties.headers.get("cam_id")
       reason = properties.headers.get("reason")
       img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
       # Execute your custom downstream YOLO or analytics model here
       ch.basic_ack(delivery_tag=method.delivery_tag)

   ch.basic_consume("frames", on_message_callback=callback)
   print("Listening for frames...")
   ch.start_consuming()
   ```

---

## Configuration Reference

| Flag | Default | Description |
|---|---|---|
| `-src` | `""` | Single source: path to video file or RTSP stream URL |
| `-cam` | `"cam1"` | Camera identifier name |
| `-config` | `""` | Path to multi-camera JSON config file |
| `-fps` | `15` | Capture frame rate |
| `-width` | `640` | Processed frame width (in pixels) |
| `-tau` | `25` | Pixel brightness difference threshold (0–255) to count as change |
| `-theta` | `0.01` | Motion threshold: fraction of changed pixels needed (0.01 = 1%) |
| `-step` | `2` | Downsample step (compares every Nth pixel for high speed) |
| `-hold` | `2.0` | Seconds to keep passing frames after motion stops |
| `-iou` | `0.15` | Minimum box overlap (IoU) for tracker continuity |
| `-loop` | `false` | Loop video files indefinitely |
| `-addr` | `":8080"` | HTTP address for the web dashboard |
| `-mp` | `""` | MediaPipe service URL (e.g. `http://localhost:8765`) |
| `-mpevery` | `1` | Run MediaPipe every Nth frame (1 = all frames, 2 = alternate) |
| `-face` | `false` | Enable face-only best-shot filtering and tracking |
| `-minshot` | `0.20` | Minimum quality score floor for best-shot candidate |
| `-bestevery`| `10` | Seconds before re-sending best shot for lingering targets |
| `-rec` | `""` | Recognition service URL (e.g. `http://localhost:8766`) |
| `-queue` | `"mem"` | Queue backend: `mem` (in-memory) or `rabbit` (RabbitMQ) |
| `-qcap` | `500` | In-memory frame queue capacity |
| `-save` | `""` | Optional folder to save all queued JPEGs to disk |

---

## Best-Shot Quality Scoring Deep-Dive

Motiongate's best-shot selection engine evaluates every frame a person is visible using a zero-trust multi-factor algorithm:

1. **Face Confidence Gate:** $G_{\text{conf}} = \text{Score}^{1.8}$
2. **Frontality Verification:** Eye landmarks must be horizontally separated and centered relative to the head bounding box.
3. **Sharpness Calculation:** Uses Laplacian variance ($\sigma^2_{\Delta}$) over the cropped grayscale face patch.
4. **Lighting & Luma Check:** Evaluates mean luma (penalizing over-exposed $> 200$ and under-exposed $< 60$ regions).
5. **Ceiling-Angle Penalty:** Penalizes top-down ceiling camera angles where head-to-body vertical ratio exceeds $0.40$.

Only frames passing quality criteria are retained, ensuring that the biometric matcher receives pristine, high-resolution frontal crops.

---

## Repository Structure

```
.
├── camera.go                   # Camera pipeline, best-shot candidate accumulation & scoring
├── cameras.example.json        # Multi-camera configuration template
├── data/                       # Local SQLite DB & audit crops (ignored in git)
├── enroll.py                   # CLI tool to enroll faces into the recognition database
├── enroll_photos/              # Enrollment image directory
├── go.mod                      # Go module definition (Go 1.22+)
├── main.go                     # Entrypoint, flag parsing, camera coordination & web server
├── make_test_video.sh          # Helper script to generate synthetic test MP4s via ffmpeg
├── mediapipe_service.py        # Python HTTP service: EfficientDet + BlazeFace detection
├── motion.go                   # 3-frame downsampled temporal difference motion detector
├── personclient.go             # Go client for MediaPipe HTTP service
├── queue.go                    # In-memory ring buffer queue interface
├── queue_rabbit.go             # RabbitMQ queue integration (build tag: rabbitmq)
├── queue_rabbit_stub.go        # Stub when RabbitMQ tag is disabled
├── recognizer.go               # Go client for InsightFace recognition service
├── recognizer_service.py       # Python HTTP service: InsightFace ArcFace + SQLite matcher
├── requirements.txt            # Python dependencies for MediaPipe service
├── requirements-rec.txt        # Python dependencies for InsightFace recognition service
├── source.go                   # ffmpeg subprocess streaming & frame decoder
├── tracker.go                  # Centroid & IoU track association engine
├── web.go                      # Embedded HTTP server & SSE hub
├── web/
│   └── index.html              # Vanilla JS/CSS live streaming dashboard
└── worker.go                   # Consumer worker, downstream dispatcher & audit logger
```

---

## License

MIT License. Free for personal and commercial use.
