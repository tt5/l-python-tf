#!/usr/bin/env python3
"""test_cvae2.py

Test the mnist2 classifier (ONNX) on 22x22 PBM images from telephonegame/data/pbm_test,
and generation quality of cvae2_generator.onnx using mnist2_model.onnx as judge.

Mirrors mnist2.py dataset handling: upscales to 28x28, derives labels including low_conf.

Usage:
    python test_cvae2.py
"""

import argparse
import numpy as np
from pathlib import Path
import onnxruntime as ort
import tensorflow as tf

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PBM_DIR = SCRIPT_DIR.parent.parent / "telephonegame" / "data" / "pbm_test"
DEFAULT_CLS_MODEL = SCRIPT_DIR / "mnist2_model.onnx"
DEFAULT_GEN_MODEL = SCRIPT_DIR / "cvae2_generator.onnx"

TARGET_SIZE = 28
LOW_CONF_THRESHOLD = 0.7
SAMPLES_PER_CLASS = 5000


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
    parser = argparse.ArgumentParser(description="Test mnist2 classifier on PBM test set and generator quality")
    parser.add_argument("--pbm-dir", type=Path, default=DEFAULT_PBM_DIR, help="PBM test directory")
    parser.add_argument("--cls-model", type=Path, default=DEFAULT_CLS_MODEL, help="Classifier ONNX path")
    parser.add_argument("--gen-model", type=Path, default=DEFAULT_GEN_MODEL, help="Generator ONNX path")
    args = parser.parse_args()

    pbm_dir = args.pbm_dir
    cls_model = args.cls_model
    gen_model = args.gen_model

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
        name = str(c) if c < max_label + 1 else "low_conf"
        print(f"  {name}: {count}")
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

    preds = probs.argmax(axis=1)

    print("\n─── Per-class accuracy ───")
    total_correct = 0
    total_samples = len(labels)
    for c in range(NUM_CLASSES):
        mask = labels == c
        n = int(mask.sum())
        if n == 0:
            name = str(c) if c < max_label + 1 else "low_conf"
            print(f"  Class {name}: no data")
            continue
        correct = int((preds[mask] == c).sum())
        total_correct += correct
        name = str(c) if c < max_label + 1 else "low_conf"
        print(f"  Class {name}: {correct}/{n} = {correct/n*100:.1f}%")

    print(f"\nOverall accuracy: {total_correct}/{total_samples} = {total_correct/total_samples*100:.1f}%")

    # Generation quality test
    print(f"\n─── Generation quality ({SAMPLES_PER_CLASS} per class) ───")
    print(f"Loading generator ({gen_model})...")
    gen_session = ort.InferenceSession(str(gen_model))
    gen_input_names = [inp.name for inp in gen_session.get_inputs()]
    gen_output_name = gen_session.get_outputs()[0].name

    input_0_shape = gen_session.get_inputs()[0].shape[1]
    input_1_shape = gen_session.get_inputs()[1].shape[1]

    if input_0_shape == NUM_CLASSES:
        label_input_name = gen_input_names[0]
        latent_input_name = gen_input_names[1]
        LATENT_DIM = input_1_shape
    else:
        latent_input_name = gen_input_names[0]
        label_input_name = gen_input_names[1]
        LATENT_DIM = input_0_shape

    print(f"  Latent dim: {LATENT_DIM}")
    print(f"  Conditioning input: {label_input_name}")

    for c in range(NUM_CLASSES):
        label_oh = np.zeros((SAMPLES_PER_CLASS, NUM_CLASSES), dtype=np.float32)
        label_oh[:, c] = 1.0
        noise = np.random.normal(size=(SAMPLES_PER_CLASS, LATENT_DIM)).astype(np.float32)

        gen_inputs = {
            latent_input_name: noise,
            label_input_name: label_oh,
        }
        gen_images = gen_session.run([gen_output_name], gen_inputs)[0]

        # Generator output is (N, 28, 28, 1) per train_cvae2.py export
        gen_images = gen_images[:, :, :, 0]
        gen_images = (gen_images > 0.5).astype(np.float32)

        raw = cls_session.run(
            [cls_output_name],
            {cls_input_name: gen_images.reshape(-1, TARGET_SIZE, TARGET_SIZE)}
        )[0]
        probs = tf.nn.softmax(raw, axis=-1).numpy()
        preds = probs.argmax(axis=1)
        correct = preds == c
        accuracy = correct.mean()
        conf = float(probs[correct, c].mean()) if correct.any() else 0.0
        print(f"  Class {c}: {accuracy*100:.1f}% correctly classified ({int(correct.sum())}/{SAMPLES_PER_CLASS}), avg confidence: {conf*100:.1f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
