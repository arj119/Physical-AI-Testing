#!/usr/bin/env python3
"""Download a YOLOv5-nano ONNX model for local development testing.

This fetches the standard YOLOv5n pretrained on COCO (80 classes) and saves
it to ``./models/yolov5n.onnx``.  The inference pipeline runs end-to-end
with this model, but detections will map to ``widget_unknown`` since COCO
classes aren't in our domain.  Replace with a fine-tuned model for production.

Usage:
    python scripts/download_model.py
"""

import os
import sys
import urllib.request

# YOLOv5n ONNX export from ultralytics (official release)
MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "yolov5n.onnx")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
        print(f"Model already exists: {OUTPUT_PATH} ({size_mb:.1f} MB)")
        resp = input("Re-download? [y/N] ").strip().lower()
        if resp != "y":
            return

    print(f"Downloading YOLOv5n ONNX from:\n  {MODEL_URL}")
    print(f"Saving to:\n  {OUTPUT_PATH}")

    try:
        urllib.request.urlretrieve(MODEL_URL, OUTPUT_PATH, _progress)
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        sys.exit(1)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\nDone! {size_mb:.1f} MB saved to {OUTPUT_PATH}")
    print("\nTo use it:")
    print(f'  MODEL_PATH={OUTPUT_PATH} python -m qa_cell_edge_agent.main --mock-hardware')
    print("\nNote: COCO detections will map to 'widget_unknown'. Fine-tune")
    print("on your own dataset for widget_good / widget_defect classes.")


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        bar = "=" * (pct // 2) + ">" + " " * (50 - pct // 2)
        print(f"\r  [{bar}] {pct}%  ({downloaded // 1024} KB)", end="", flush=True)


if __name__ == "__main__":
    main()
