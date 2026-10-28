#!/usr/bin/env python3
"""MediaPipe person + face detector as a tiny local HTTP service for motiongate.
POST /detect  (body = JPEG bytes)  ->  [{x,y,w,h,score,face:{x,y,w,h,score,lx,ly,rx,ry}|null}, ...]
GET  /health
"""
import argparse, json, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientdet_lite0.tflite")
    ap.add_argument("--face-model", default="blaze_face_short_range.tflite")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--score", type=float, default=0.4)
    ap.add_argument("--face-score", type=float, default=0.5)
    ap.add_argument("--max", type=int, default=20)
    a = ap.parse_args()

    person_detector = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=a.model),
        score_threshold=a.score, max_results=a.max, category_allowlist=["person"]))

    face_detector = None
    try:
        import os
        if os.path.exists(a.face_model) and os.path.getsize(a.face_model) > 1000:
            face_detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=a.face_model),
                min_detection_confidence=a.face_score))
            print(f"Face detector loaded: {a.face_model}")
        else:
            print(f"WARNING: face model not found or empty: {a.face_model}. Face detection disabled.")
    except Exception as e:
        print(f"WARNING: could not load face model: {e}. Face detection disabled.")

    # Also try face landmarker for frontality (optional)
    face_landmarker = None
    try:
        import os
        lm_path = "face_landmarker.task"
        if os.path.exists(lm_path) and os.path.getsize(lm_path) > 1000:
            from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
            face_landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=lm_path)))
            print(f"Face landmarker loaded: {lm_path}")
    except Exception as e:
        pass  # landmarker is optional

    lock = threading.Lock()

    def detect_face_in_person(bgr_frame, px, py, pw, ph):
        """Detect face in person crop. Returns face dict or None."""
        ih, iw = bgr_frame.shape[:2]
        # head region: widen 10%, top 50% of person box
        pad_x = int(pw * 0.10)
        cx = max(0, px - pad_x)
        cw = min(iw - cx, pw + 2 * pad_x)
        cy = py
        ch = min(ih - cy, ph // 2)
        if cw < 32 or ch < 32:
            return None
        crop = bgr_frame[cy:cy+ch, cx:cx+cw]
        crop_rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        result = face_detector.detect(crop_img)
        if not result.detections:
            return None
        # pick highest score
        best = max(result.detections, key=lambda d: d.categories[0].score if d.categories else 0)
        if not best.categories:
            return None
        fb = best.bounding_box
        fscore = float(best.categories[0].score)
        # ignore if face height > 60% of person height
        if fb.height > 0.60 * ph:
            return None
        # convert to full-frame coords
        fx = max(0, min(iw - 1, int(cx + fb.origin_x)))
        fy = max(0, min(ih - 1, int(cy + fb.origin_y)))
        fw = min(iw - fx, int(fb.width))
        fh = min(ih - fy, int(fb.height))

        # Forehead & Luma Verification:
        # On a genuine face, the top of the face box is forehead skin (illuminated from above).
        # On a back-of-head fake (nape of neck), the top of the box is pitch-black hair and the bottom is neck skin.
        bx = max(0, int(fb.origin_x))
        by = max(0, int(fb.origin_y))
        bw = min(crop.shape[1] - bx, int(fb.width))
        bh = min(crop.shape[0] - by, int(fb.height))
        if bw >= 16 and bh >= 16:
            face_patch = crop[by:by+bh, bx:bx+bw]
            ycrcb = cv2.cvtColor(face_patch, cv2.COLOR_BGR2YCrCb)
            cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
            skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
            overall_skin = float(np.mean(skin_mask))
            # Real faces have 15-50% facial skin; back of head/hair has 0-2%
            if overall_skin < 0.08:
                return None

            top_half = ycrcb[:bh//2, :]
            bot_half = ycrcb[bh//2:, :]
            top_skin = np.mean((top_half[:, :, 1] >= 133) & (top_half[:, :, 1] <= 173) & (top_half[:, :, 2] >= 77) & (top_half[:, :, 2] <= 127))
            top_luma = float(np.mean(top_half[:, :, 0]))
            bot_luma = float(np.mean(bot_half[:, :, 0]))
            luma_ratio = top_luma / (bot_luma + 1e-5)
            # If top is dark hair while bottom is brightly lit neck skin -> discard hair/neck pareidolia
            if top_skin < 0.35 and luma_ratio < 0.68:
                return None

        # Extract keypoints from BlazeFace: 0=r_eye, 1=l_eye, 2=nose, 3=mouth, 4=r_ear, 5=l_ear
        lx, ly, rx, ry = None, None, None, None
        if hasattr(best, 'keypoints') and best.keypoints and len(best.keypoints) >= 4:
            kps = best.keypoints
            rx, ry = float(kps[0].x), float(kps[0].y)
            lx, ly = float(kps[1].x), float(kps[1].y)
            nx, ny = float(kps[2].x), float(kps[2].y)
            mx, my = float(kps[3].x), float(kps[3].y)

            # Facial anatomy verification:
            # 1. Eyes must have horizontal separation
            # 2. Eyes must be vertically above the nose
            # 3. Nose must be vertically above the mouth
            eyes_y = (ry + ly) / 2.0
            eye_dist = abs(lx - rx)
            if eye_dist < 0.03 or ny < eyes_y or my < ny:
                return None

        face = {"x": fx, "y": fy, "w": fw, "h": fh, "score": round(fscore, 3)}
        if lx is not None:
            face["lx"] = round(lx, 4)
            face["ly"] = round(ly, 4)
            face["rx"] = round(rx, 4)
            face["ry"] = round(ry, 4)
        return face

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, {"ok": True}) if self.path == "/health" else self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/detect":
                return self._send(404, {"error": "not found"})
            data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                return self._send(400, {"error": "bad jpeg"})
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            with lock:
                result = person_detector.detect(image)
                out = []
                for d in result.detections:
                    b = d.bounding_box
                    px, py = max(0, int(b.origin_x)), max(0, int(b.origin_y))
                    pw, ph = int(b.width), int(b.height)
                    person_score = round(float(d.categories[0].score), 3)
                    face = None
                    if face_detector is not None:
                        face = detect_face_in_person(bgr, px, py, pw, ph)
                    out.append({"x": px, "y": py, "w": pw, "h": ph, "score": person_score, "face": face})
            self._send(200, out)

        def log_message(self, *args):
            pass

    print(f"MediaPipe person+face service on http://localhost:{a.port}  (person model {a.model}, face model {a.face_model})")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()

if __name__ == "__main__":
    main()
