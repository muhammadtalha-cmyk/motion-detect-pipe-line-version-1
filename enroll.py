#!/usr/bin/env python3
"""Enrolment script for motiongate face recognition service.
Uses only Python standard library.
Usage:
    python enroll.py --dir enroll_photos [--url http://localhost:8766]
    python enroll.py --list [--url http://localhost:8766]
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description="Enrol face photos into motiongate recognition service")
    parser.add_argument("--dir", default="enroll_photos", help="Directory with subfolders per person")
    parser.add_argument("--url", default="http://localhost:8766", help="Recognition service URL")
    parser.add_argument("--list", action="store_true", help="List enrolled people")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    if args.list:
        req = urllib.request.Request(f"{base_url}/people", method="GET")
        try:
            with urllib.request.urlopen(req) as resp:
                people = json.loads(resp.read().decode("utf-8"))
                print(f"Enrolled people ({len(people)}):")
                for p in people:
                    print(f"  - {p['name']} ({p.get('embeddings', 0)} embedding(s))")
        except Exception as e:
            print(f"Error connecting to recognition service at {base_url}: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if not os.path.isdir(args.dir):
        print(f"Directory not found: {args.dir}", file=sys.stderr)
        sys.exit(1)

    accepted = 0
    skipped = 0
    people = set()

    valid_exts = {".jpg", ".jpeg", ".png"}
    items_to_enroll = []

    for entry in sorted(os.listdir(args.dir)):
        if entry.startswith(".") or entry == "README.txt":
            continue
        full_path = os.path.join(args.dir, entry)
        if os.path.isdir(full_path):
            person_name = entry
            files = [f for f in sorted(os.listdir(full_path)) if os.path.splitext(f)[1].lower() in valid_exts]
            for filename in files:
                items_to_enroll.append((person_name, filename, os.path.join(full_path, filename)))
        elif os.path.isfile(full_path):
            stem, ext = os.path.splitext(entry)
            if ext.lower() in valid_exts:
                items_to_enroll.append((stem, entry, full_path))

    for person_name, filename, filepath in items_to_enroll:
        try:
            with open(filepath, "rb") as f:
                img_bytes = f.read()
        except Exception as e:
            print(f"[SKIPPED] {person_name}/{filename}: cannot read file ({e})")
            skipped += 1
            continue

        query = urllib.parse.urlencode({"name": person_name, "file": filename})
        req = urllib.request.Request(
            f"{base_url}/enroll?{query}",
            data=img_bytes,
            headers={"Content-Type": "image/jpeg"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                print(f"[OK] {person_name} (from {filename}, embeddings: {data.get('embeddings')}, det_score: {data.get('det_score')})")
                accepted += 1
                people.add(person_name)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            try:
                err_json = json.loads(err_body)
                reason = err_json.get("error", err_body)
            except Exception:
                reason = err_body
            print(f"[SKIPPED] {person_name}/{filename}: HTTP {e.code} ({reason})")
            skipped += 1
        except Exception as e:
            print(f"[SKIPPED] {person_name}/{filename}: {e}")
            skipped += 1

    print(f"\nSummary: {len(people)} people, {accepted} photos accepted, {skipped} photos skipped.")


if __name__ == "__main__":
    main()
