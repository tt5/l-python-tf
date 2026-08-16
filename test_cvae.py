#!/usr/bin/env python3
"""test_cvae.py

Test the CVAE generator (ONNX) after training.
Evaluates generation quality by sampling from real data.

Usage:
    python test_cvae.py
"""

import onnxruntime as ort
import numpy as np
import argparse
from pathlib import Path
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--num-classes", type=int, default=10, help="Number of classes to evaluate (default: 10)")
args = parser.parse_args()
NUM_CLASSES = args.num_classes

save_dir = Path(__file__).resolve().parent


# ─── Load models ────────────────────────────────────────────────────
print("Loading generator (ONNX)...")
gen_session = ort.InferenceSession(str(save_dir / "cvae_generator.onnx"))
gen_input_names = [inp.name for inp in gen_session.get_inputs()]
gen_output_name = gen_session.get_outputs()[0].name

label_input_name = next(n for n in gen_input_names if 'label_input' in n)
latent_input_name = next(n for n in gen_input_names if 'latent_input' in n)

label_input_info = gen_session.get_inputs()[gen_input_names.index(label_input_name)]
latent_input_info = gen_session.get_inputs()[gen_input_names.index(latent_input_name)]
NUM_CLASSES = int(label_input_info.shape[1])
LATENT_DIM = int(latent_input_info.shape[1])
print(f"  Inputs: {gen_input_names}")
print(f"  NUM_CLASSES: {NUM_CLASSES}")
print(f"  LATENT_DIM: {LATENT_DIM}")
print(f"  Output: {gen_output_name}")

print("Loading mnist classifier (ONNX)...")
cls_session = ort.InferenceSession(str(save_dir / "mnist_model.onnx"))
cls_input_name = cls_session.get_inputs()[0].name
cls_output_name = cls_session.get_outputs()[0].name

SAMPLES_PER_CLASS = 1000
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic" / "test_data"

# ─── Generation quality ─────────────────────────────────────────────
print(f"\n─── Sample generation quality ({SAMPLES_PER_CLASS} per class) ───")

for c in range(NUM_CLASSES):
    label_oh = np.zeros((SAMPLES_PER_CLASS, NUM_CLASSES), dtype=np.float32)
    label_oh[:, c] = 1.0

    noise = np.random.normal(size=(SAMPLES_PER_CLASS, LATENT_DIM)).astype(np.float32)

    gen_inputs = {
        latent_input_name: noise,
        label_input_name: label_oh,
    }
    gen_images = gen_session.run([gen_output_name], gen_inputs)[0][:, :, :, 0]
    gen_images = (gen_images > 0.5).astype(np.float32)

    raw = cls_session.run(
        [cls_output_name],
        {cls_input_name: gen_images.reshape(-1, 28, 28, 1).astype(np.float32)}
    )[0]
    probs = tf.nn.softmax(raw, axis=-1).numpy()
    preds = probs.argmax(axis=1)
    correct = preds == c
    accuracy = correct.mean()
    conf = float(probs[correct, c].mean()) if correct.any() else 0.0
    print(f"  Class {c}: {accuracy*100:.1f}% correctly classified ({int(correct.sum())}/{SAMPLES_PER_CLASS}), avg confidence: {conf*100:.1f}%")

# ─── Real data classification ────────────────────────────────────────
print(f"\n─── Real data classification (from synthetic project) ───")

for c in range(NUM_CLASSES - 1):
    class_dir = DATA_DIR / f"class_{c}"
    files = sorted(class_dir.glob("*.npy"))
    if not files:
        print(f"  Class {c}: no data found")
        continue

    n_samples = min(len(files), SAMPLES_PER_CLASS)
    images = []
    for f in files[:n_samples]:
        img = np.load(str(f)).astype(np.float32) / 255.0
        img = (img > 0.5).astype(np.float32)
        images.append(img)
    images = np.array(images).reshape(-1, 28, 28, 1).astype(np.float32)

    preds = cls_session.run([cls_output_name], {cls_input_name: images})[0]
    preds = preds.argmax(axis=1)
    accuracy = (preds == c).mean()
    print(f"  Class {c}: {accuracy*100:.1f}% correctly classified ({int((preds == c).sum())}/{n_samples})")

# ─── Synthetic last-class evaluation ─────────────────────────────────
print(f"\n─── Synthetic class evaluation (generated on-the-fly) ───")
synthetic_label = NUM_CLASSES - 1
flip_pixels = 7 * 28
pixel_indices = np.arange(28 * 28)
images = []
for _ in range(SAMPLES_PER_CLASS):
    sample = np.zeros((28, 28), dtype=np.float32)
    flips = np.random.choice(pixel_indices, size=np.random.randint(3, flip_pixels), replace=False)
    sample.reshape(-1)[flips] = 1.0
    images.append(sample)
images = np.array(images).reshape(-1, 28, 28, 1).astype(np.float32)

preds = cls_session.run([cls_output_name], {cls_input_name: images})[0]
preds = preds.argmax(axis=1)
accuracy = (preds == synthetic_label).mean()
print(f"  Class {synthetic_label}: {accuracy*100:.1f}% correctly classified ({int((preds == synthetic_label).sum())}/{SAMPLES_PER_CLASS})")

print("\nDone!")
