#!/usr/bin/env python3
"""test_cvae2.py

Test the mnist2 classifier (ONNX) on 22x22 PBM images from telephonegame/data/pbm_test.
Mirrors mnist2.py dataset handling: upscales to 28x28, derives labels including low_conf.

Usage:
    python test_cvae2.py
"""

import argparse
import numpy as np
from pathlib import Path
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PBM_DIR = SCRIPT_DIR.parent.parent / "telephonegame" / "data" / "pbm_test"
DEFAULT_CLS_MODEL = SCRIPT_DIR / "mnist2_model.onnx"

TARGET_SIZE = 28
LOW_CONF_THRESHOLD = 0.6


def load_pbm(path):
    """Load a P4 PBM file and return a 2D numpy array of 0/1 float32."""
    data = path.read_bytes()
    first_nl = data.index(b'\n')
    second_nl = data.index(b'\n', first_nl + 1)
    header = data[:second_nl].decode('ascii').strip()
    parts = header.split('\n')
    dims = parts[1].split()
    width, height = int(dims[0]), int(dims[1])
    pixel_data = data[second_nl + 1:]

    grid = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            byte_idx = y * ((width + 7) // 8) + x // 8
            bit = (pixel_data[byte_idx] >> (7 - x % 8)) & 1
            grid[y, x] = 1.0 - bit
    return grid


def upscale(grid, dst_h, dst_w):
    """Nearest-neighbor block upscale."""
    src_h, src_w = grid.shape
    out = np.zeros((dst_h, dst_w), dtype=np.float32)
    base_h, extra_h = divmod(dst_h, src_h)
    base_w, extra_w = divmod(dst_w, src_w)
    for r in range(src_h):
        for c in range(src_w):
            r_start = r * base_h + min(r, extra_h)
            c_start = c * base_w + min(c, extra_w)
            r_end = min(r_start + base_h + (1 if r < extra_h else 0), dst_h)
            c_end = min(c_start + base_w + (1 if c < extra_w else 0), dst_w)
            out[r_start:r_end, c_start:c_end] = grid[r, c]
    return out


def main():
    parser = argparse.ArgumentParser(description="Test mnist2 classifier on PBM test set")
    parser.add_argument("--pbm-dir", type=Path, default=DEFAULT_PBM_DIR, help="PBM test directory")
    parser.add_argument("--cls-model", type=Path, default=DEFAULT_CLS_MODEL, help="Classifier ONNX path")
    args = parser.parse_args()

    pbm_dir = args.pbm_dir
    cls_model = args.cls_model

    files = sorted(pbm_dir.glob("*.pbm"))
    print(f"Loading {len(files)} PBM images from {pbm_dir}...")

    if not files:
        print("No PBM files found.")
        return

    images = []
    labels = []
    low_conf_count = 0

    for i, f in enumerate(files):
        parts = f.stem.split('_')
        predicted_digit = int(parts[0])
        confidence = float(parts[1]) / 10000.0

        img = load_pbm(f)
        img_28 = upscale(img, TARGET_SIZE, TARGET_SIZE)
        img_28 = (img_28 > 0.5).astype(np.float32)

        images.append(img_28)

        if confidence < LOW_CONF_THRESHOLD:
            labels.append(-1)  # placeholder, will be remapped
            low_conf_count += 1
        else:
            labels.append(predicted_digit)

    images = np.array(images)
    labels = np.array(labels)

    # Derive NUM_CLASSES same as mnist2.py
    max_label = labels[labels >= 0].max()
    low_conf_label = max_label + 1
    labels[labels == -1] = low_conf_label
    NUM_CLASSES = low_conf_label + 1

    print(f"Derived NUM_CLASSES = {NUM_CLASSES} (max_label={max_label}, low_conf={low_conf_label})")
    print(f"\nLabel distribution (low_conf threshold={LOW_CONF_THRESHOLD}):")
    for c in range(NUM_CLASSES):
        count = int(np.sum(labels == c))
        print(f"  {c if c < max_label + 1 else 'low_conf'}: {count}")
    print(f"  low_conf_count={low_conf_count}")

    # Load classifier
    print(f"\nLoading classifier ({cls_model})...")
    cls_session = ort.InferenceSession(str(cls_model))
    cls_input_name = cls_session.get_inputs()[0].name
    cls_output_name = cls_session.get_outputs()[0].name

    raw = cls_session.run(
        [cls_output_name],
        {cls_input_name: images}
    )[0]
    probs = raw  # logits expected for mnist2 model

    # mnist2 model input is (N, 28, 28), output logits (N, NUM_CLASSES)
    preds = probs.argmax(axis=1)

    print("\n─── Per-class accuracy ───")
    total_correct = 0
    total_samples = len(labels)
    for c in range(NUM_CLASSES):
        mask = labels == c
        n = int(mask.sum())
        if n == 0:
            print(f"  Class {c if c < max_label + 1 else 'low_conf'}: no data")
            continue
        correct = int((preds[mask] == c).sum())
        total_correct += correct
        print(f"  Class {c if c < max_label + 1 else 'low_conf'}: {correct}/{n} = {correct/n*100:.1f}%")

    print(f"\nOverall accuracy: {total_correct}/{total_samples} = {total_correct/total_samples*100:.1f}%")
    print("\nDone!")


if __name__ == "__main__":
    main()
