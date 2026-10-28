#!/usr/bin/env python3
"""InsightFace recognition service for motiongate.
Endpoints:
    GET  /health
    GET  /people
    POST /enroll?name=<Name>&file=<filename>
    POST /recognize?cam=<cam>&track=<int>&seq=<int>
"""
import argparse
from datetime import datetime, timezone
import json
import os
import sqlite3
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Motiongate Face Recognition Service")
    parser.add_argument("--port", type=int, default=8766, help="Port to listen on (default: 8766)")
    parser.add_argument("--db", default="data/faces.db", help="Path to SQLite database (default: data/faces.db)")
    parser.add_argument("--model", default="buffalo_l", help="InsightFace model pack name (default: buffalo_l)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Cosine similarity threshold for match")
    parser.add_argument("--margin", type=float, default=0.05, help="Margin over 2nd best score")
    parser.add_argument("--min-det", type=float, default=0.50, help="Minimum detection confidence")
    parser.add_argument("--min-face", type=int, default=20, help="Minimum face width in pixels")
    args = parser.parse_args()

    # Ensure DB directory exists
    db_dir = os.path.dirname(os.path.abspath(args.db))
    os.makedirs(db_dir, exist_ok=True)

    print(f"Connecting to database: {args.db}")
    db = sqlite3.connect(args.db, check_same_thread=False)

    # Initialize tables
    with db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL REFERENCES people(id),
                emb BLOB NOT NULL,
                source_file TEXT,
                det_score REAL,
                created_at TEXT NOT NULL
            );
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS recognitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                cam_id TEXT,
                track_id INTEGER,
                seq INTEGER,
                status TEXT NOT NULL,
                person_id INTEGER,
                name TEXT,
                similarity REAL,
                second_best REAL,
                det_score REAL,
                crop_path TEXT
            );
        """)

    print(f"Loading InsightFace model: {args.model}...")
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=args.model, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("InsightFace model loaded successfully.")

    # Global lock around model inference and DB operations
    lock = threading.Lock()

    # In-memory matrix M and owner mappings
    # M: shape (N, 512), float32
    # owner_ids: list of person_ids for each row
    # person_names: dict mapping person_id -> name
    state = {
        "M": np.empty((0, 512), dtype=np.float32),
        "owner_ids": [],
        "person_names": {}
    }

    def rebuild_matrix():
        cur = db.cursor()
        cur.execute("SELECT id, name FROM people")
        person_names = {row[0]: row[1] for row in cur.fetchall()}

        cur.execute("SELECT person_id, emb FROM embeddings ORDER BY id ASC")
        rows = cur.fetchall()

        if rows:
            embs = []
            owner_ids = []
            for pid, blob in rows:
                vec = np.frombuffer(blob, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                embs.append(vec)
                owner_ids.append(pid)
            state["M"] = np.stack(embs, axis=0).astype(np.float32)
            state["owner_ids"] = owner_ids
        else:
            state["M"] = np.empty((0, 512), dtype=np.float32)
            state["owner_ids"] = []
        state["person_names"] = person_names
        print(f"Index rebuilt: {len(state['person_names'])} people, {len(state['owner_ids'])} embeddings.")

    with lock:
        rebuild_matrix()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                with lock:
                    n_people = len(state["person_names"])
                    n_embs = len(state["owner_ids"])
                return self._send(200, {"ok": True, "people": n_people, "embeddings": n_embs})

            elif parsed.path == "/people":
                with lock:
                    cur = db.cursor()
                    cur.execute("""
                        SELECT p.name, COUNT(e.id) AS cnt
                        FROM people p
                        LEFT JOIN embeddings e ON p.id = e.person_id
                        GROUP BY p.id, p.name
                        ORDER BY p.name ASC
                    """)
                    rows = cur.fetchall()
                out = [{"name": r[0], "embeddings": r[1]} for r in rows]
                return self._send(200, out)

            else:
                return self._send(404, {"error": "not found"})

        def do_POST(self):
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)

            if parsed.path == "/enroll":
                name = qs.get("name", [None])[0]
                filename = qs.get("file", ["image.jpg"])[0]
                if not name:
                    return self._send(400, {"error": "missing 'name' parameter"})

                length = int(self.headers.get("Content-Length", 0))
                data = self.rfile.read(length)
                img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    return self._send(400, {"error": "bad image"})

                with lock:
                    faces = app.get(img_bgr)
                    if not faces:
                        return self._send(422, {"ok": False, "error": "no face found"})

                    # Pick largest face by bbox area
                    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    det_score = float(best_face.det_score)
                    if det_score < 0.60:
                        return self._send(422, {"ok": False, "error": f"det_score {det_score:.3f} < 0.60"})

                    # Extract and normalize embedding
                    e = best_face.normed_embedding.astype(np.float32)
                    norm = np.linalg.norm(e)
                    if norm > 0:
                        q = e / norm
                    else:
                        q = e

                    now = datetime.now(timezone.utc).isoformat()
                    cur = db.cursor()
                    cur.execute("INSERT OR IGNORE INTO people (name, created_at) VALUES (?, ?)", (name, now))
                    cur.execute("SELECT id FROM people WHERE name = ?", (name,))
                    person_id = cur.fetchone()[0]

                    cur.execute(
                        "INSERT INTO embeddings (person_id, emb, source_file, det_score, created_at) VALUES (?, ?, ?, ?, ?)",
                        (person_id, q.astype(np.float32).tobytes(), filename, det_score, now)
                    )
                    db.commit()

                    rebuild_matrix()

                    cur.execute("SELECT COUNT(*) FROM embeddings WHERE person_id = ?", (person_id,))
                    emb_count = cur.fetchone()[0]

                return self._send(200, {
                    "ok": True,
                    "name": name,
                    "embeddings": emb_count,
                    "det_score": round(det_score, 3)
                })

            elif parsed.path == "/recognize":
                t0 = time.time()
                cam = qs.get("cam", ["unknown"])[0]
                try:
                    track_id = int(qs.get("track", [0])[0])
                except ValueError:
                    track_id = 0
                try:
                    seq = int(qs.get("seq", [0])[0])
                except ValueError:
                    seq = 0

                length = int(self.headers.get("Content-Length", 0))
                data = self.rfile.read(length)
                img_bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    return self._send(400, {"error": "bad image"})

                now_dt = datetime.now(timezone.utc)
                ts_str = now_dt.isoformat()
                date_folder = now_dt.strftime("%Y%m%d")
                shot_dir = os.path.join("data", "shots", date_folder)
                os.makedirs(shot_dir, exist_ok=True)
                crop_rel_path = os.path.join(shot_dir, f"{cam}_t{track_id}_{seq}.jpg")

                with open(crop_rel_path, "wb") as f_out:
                    f_out.write(data)

                status = "unknown"
                matched_id = None
                matched_name = None
                s1_val = None
                s2_val = None
                closest_name = None
                det_score_val = None
                face_px_val = None

                with lock:
                    faces = app.get(img_bgr)
                    if not faces:
                        status = "no_face"
                    else:
                        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                        det_score_val = float(best_face.det_score)
                        face_width = float(best_face.bbox[2] - best_face.bbox[0])
                        face_px_val = int(round(face_width))

                        if det_score_val < args.min_det or face_width < args.min_face:
                            status = "low_quality"
                        elif len(state["owner_ids"]) == 0:
                            status = "unknown"
                            closest_name = None
                        else:
                            e = best_face.normed_embedding.astype(np.float32)
                            norm = np.linalg.norm(e)
                            q = e / norm if norm > 0 else e

                            sims = state["M"] @ q  # shape (N,)
                            # Group by person_id -> max sim
                            person_sims = {}
                            for i, pid in enumerate(state["owner_ids"]):
                                score = float(sims[i])
                                if pid not in person_sims or score > person_sims[pid]:
                                    person_sims[pid] = score

                            # Sort people descending by similarity
                            sorted_people = sorted(person_sims.items(), key=lambda x: x[1], reverse=True)

                            p1_id, S1 = sorted_people[0]
                            s1_val = S1
                            p1_name = state["person_names"].get(p1_id, "unknown")

                            if len(sorted_people) > 1:
                                p2_id, S2 = sorted_people[1]
                                s2_val = S2
                            else:
                                S2 = None

                            if S1 >= args.threshold and (S2 is None or (S1 - S2) >= args.margin):
                                status = "match"
                                matched_id = p1_id
                                matched_name = p1_name
                            else:
                                status = "unknown"
                                closest_name = p1_name

                    # Save recognition to database
                    sim_rep = round(s1_val, 3) if s1_val is not None else None
                    sec_rep = round(s2_val, 3) if s2_val is not None else None
                    det_rep = round(det_score_val, 3) if det_score_val is not None else None

                    cur = db.cursor()
                    cur.execute("""
                        INSERT INTO recognitions (
                            ts, cam_id, track_id, seq, status, person_id, name,
                            similarity, second_best, det_score, crop_path
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        ts_str, cam, track_id, seq, status, matched_id, matched_name,
                        sim_rep, sec_rep, det_rep, crop_rel_path
                    ))
                    rec_id = cur.lastrowid
                    db.commit()

                elapsed_ms = (time.time() - t0) * 1000.0
                print(f"[{cam}] track={track_id} seq={seq} -> status={status} name={matched_name} sim={sim_rep} ({elapsed_ms:.1f}ms)")

                return self._send(200, {
                    "id": rec_id,
                    "status": status,
                    "name": matched_name,
                    "similarity": sim_rep,
                    "second": sec_rep,
                    "closest": closest_name,
                    "det_score": det_rep,
                    "face_px": face_px_val
                })

            else:
                return self._send(404, {"error": "not found"})

        def log_message(self, *args):
            # Suppress default HTTP request logging to keep output clean
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Face recognition service listening on http://127.0.0.1:{args.port} (database: {args.db})")
    server.serve_forever()


if __name__ == "__main__":
    main()
