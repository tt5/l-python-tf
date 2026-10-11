#!/usr/bin/env python3
"""test_mnist.py

Evaluate the trained mnist.py classifier on the test split and report
where the errors live: overall accuracy, per-class accuracy (worst
class first), and a full confusion matrix.

Uses the exact same loader and preprocessing as mnist.py
(utils.mnist_loader.load_data), including the synthetic last class,
so the test split is consistent with training.

Usage:
    /home/n/miniconda3/envs/tf/bin/python test_mnist.py [--num-classes 5]
        [--model mnist_model] [--data-dir DIR] [--max-samples-per-class N]
"""

import numpy as np
import argparse
from pathlib import Path

import tensorflow as tf

from utils.mnist_loader import load_data

parser = argparse.ArgumentParser()
parser.add_argument("--num-classes", type=int, default=5)
parser.add_argument("--model", type=str, default="mnist_model",
                    help="Saved Keras model directory (default: mnist_model)")
parser.add_argument("--data-dir", type=str, default=None,
                    help="Dataset dir (default: ../synthetic/data, same as mnist.py)")
parser.add_argument("--max-samples-per-class", type=int, default=None,
                    help="Cap samples per class (speed up eval)")
parser.add_argument("--pbm-dir", type=str, default=None,
                    help="Evaluate on P4 PBM images from this directory instead")
args = parser.parse_args()

NUM_CLASSES = args.num_classes

DATA_DIR = (Path(args.data_dir) if args.data_dir
            else Path(__file__).resolve().parent.parent.parent / "synthetic" / "data")
PBM_DIR = Path(args.pbm_dir) if args.pbm_dir else None

# ─── Load data (same as mnist.py) ───────────────────────────────────
(x_train, y_train), (x_test, y_test) = load_data(
    DATA_DIR, NUM_CLASSES, flip_pixels=7 * 28,
    max_samples_per_class=args.max_samples_per_class, pbm_dir=PBM_DIR)

print(f"\nTest set: {len(x_test)} images, classes 0..{NUM_CLASSES - 1}")

# ─── Load model & evaluate ──────────────────────────────────────────
model = tf.keras.models.load_model(args.model, compile=False)
preds = model.predict(x_test, batch_size=256, verbose=0)
pred_labels = np.argmax(preds, axis=1)

correct = (pred_labels == y_test)
overall = correct.mean()

print(f"\n═══ Overall test accuracy: {overall * 100:.2f}% ═══\n")

# ─── Per-class accuracy ─────────────────────────────────────────────
print("Per-class accuracy:")
per_class = []
for c in range(NUM_CLASSES):
    mask = (y_test == c)
    n = int(mask.sum())
    acc = float(correct[mask].mean()) if n > 0 else float("nan")
    per_class.append((acc, c, n))
    print(f"  class {c}: {acc * 100:6.2f}%  ({int(correct[mask].sum())}/{n})")

per_class_valid = [(a, c, n) for a, c, n in per_class if n > 0]
worst_acc, worst_cls, worst_n = min(per_class_valid)
print(f"\nWorst class: {worst_cls} ({worst_acc * 100:.2f}%)")

# ─── Confusion matrix (rows = true, cols = predicted) ───────────────
conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
for t, p in zip(y_test, pred_labels):
    conf[t, p] += 1

print("\nConfusion matrix (rows = true class, cols = predicted):")
header = "true\\pred " + " ".join(f"{c:>6d}" for c in range(NUM_CLASSES))
print(header)
for c in range(NUM_CLASSES):
    row = " ".join(f"{conf[c, p]:>6d}" for p in range(NUM_CLASSES))
    print(f"   {c:>5d}   {row}")

# ─── Top confusions (what the model mixes up) ───────────────────────
print("\nTop confusions (true -> predicted: count):")
pairs = []
for t in range(NUM_CLASSES):
    for p in range(NUM_CLASSES):
        if t != p and conf[t, p] > 0:
            pairs.append((conf[t, p], t, p))
pairs.sort(reverse=True)
if not pairs:
    print("  (none — perfect test accuracy)")
else:
    for count, t, p in pairs[:10]:
        pct = 100.0 * count / max(1, conf[t].sum())
        print(f"  {t} -> {p}: {count}  ({pct:.1f}% of true class {t})")

print("\nDone.")
