#!/usr/bin/env python3
"""mnist3.py

Train a classifier on 22x22 binary PBM images from data/pbm2/.
Upscales to 28x28 using nearest-neighbor (same as pbm_utils.upscale),
binarizes, and trains a 12-class classifier.

Labels 0-10: predicted digit from filename
Label 11: low confidence (confidence < 0.6)

Usage:
    /home/n/miniconda3/envs/tf/bin/python mnist3.py
"""

import argparse
import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

DEFAULT_PBM_DIR = Path(__file__).resolve().parent / "../../telephonegame/data/pbm2"

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=DEFAULT_PBM_DIR, help="PBM data directory")
args, _ = parser.parse_known_args()
PBM_DIR = args.data_dir
IMG_SIZE = 22
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
            grid[y, x] = 1.0 - bit  # invert: P4 black=1 -> white digit=1
    return grid

def upscale(grid, dst_h, dst_w):
    """Nearest-neighbor block upscale (same as pbm_utils.upscale)."""
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
    files = sorted(PBM_DIR.glob("*.pbm"))
    print(f"Loading {len(files)} PBM images from {PBM_DIR}...")

    images = []
    labels = []
    low_conf_count = 0

    for i, f in enumerate(files):
        parts = f.stem.split('_')
        predicted_digit = int(parts[0])
        confidence = float(parts[1]) / 10000.0

        img = load_pbm(f)
        img_28 = upscale(img, TARGET_SIZE, TARGET_SIZE)
        # Binarize (should already be binary from upscale, but ensure)
        img_28 = (img_28 > 0.5).astype(np.float32)

        images.append(img_28)

        if confidence < LOW_CONF_THRESHOLD:
            labels.append(-1)  # placeholder, will be remapped
            low_conf_count += 1
        else:
            labels.append(predicted_digit)

        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1}/{len(files)}")

    images = np.array(images)
    labels = np.array(labels)

    # Derive num_classes from data: max label + 1 for low_conf
    max_label = labels[labels >= 0].max()
    low_conf_label = max_label + 1
    labels[labels == -1] = low_conf_label
    NUM_CLASSES = low_conf_label + 1

    print(f"\nDerived NUM_CLASSES = {NUM_CLASSES} (max_label={max_label}, low_conf={low_conf_label})")

    print(f"\nLabel distribution (low_conf threshold={LOW_CONF_THRESHOLD}):")
    for c in range(NUM_CLASSES):
        count = np.sum(labels == c)
        if c < NUM_CLASSES - 1:
            name = f"digit_{c}"
        elif c == NUM_CLASSES:
            name = "low_conf"
        else:
            name = "low_conf_2"
        print(f"  {name}: {count}")

    # Shuffle
    np.random.seed(42)
    perm = np.random.permutation(len(images))
    images = images[perm]
    labels = labels[perm]

    # Train/test split (80/20)
    split = int(len(images) * 0.8)
    x_train, x_test = images[:split], images[split:]
    y_train, y_test = labels[:split], labels[split:]
    print(f"\nTrain: {len(x_train)}, Test: {len(x_test)}")

    # Build model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(TARGET_SIZE, TARGET_SIZE)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    print("\nModel summary:")
    model.summary()

    model.fit(x_train, y_train, epochs=20, batch_size=128,
              validation_data=(x_test, y_test))

    print("\nFinal evaluation:")
    model.evaluate(x_test, y_test, verbose=2)

    # Save
    save_path = Path(__file__).parent / "mnist3_model"
    model.save(str(save_path))
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    main()
