#!/usr/bin/env python3
"""test_cvae.py

Test the CVAE generator (ONNX) after training.
Evaluates generation quality by sampling from real data.

Usage:
    python test_cvae.py
"""

import onnxruntime as ort
import numpy as np
from pathlib import Path

NUM_CLASSES = 10

save_dir = Path(__file__).resolve().parent


# ─── Load models ────────────────────────────────────────────────────
print("Loading generator (ONNX)...")
gen_session = ort.InferenceSession(str(save_dir / "cvae_generator.onnx"))
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
print(f"  Inputs: {gen_input_names}")
print(f"  Latent dim: {LATENT_DIM}")
print(f"  Output: {gen_output_name}")

print("Loading mnist classifier (ONNX)...")
cls_session = ort.InferenceSession(str(save_dir / "mnist_model.onnx"))
cls_input_name = cls_session.get_inputs()[0].name
cls_output_name = cls_session.get_outputs()[0].name

SAMPLES_PER_CLASS = 5000
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic" / "data"

## ─── Generation quality ─────────────────────────────────────────────
#print(f"\n─── Sample generation quality ({SAMPLES_PER_CLASS} per class) ───")
#
#for c in range(NUM_CLASSES):
#    label_oh = np.zeros((SAMPLES_PER_CLASS, NUM_CLASSES), dtype=np.float32)
#    label_oh[:, c] = 1.0
#
#    noise = np.random.normal(size=(SAMPLES_PER_CLASS, LATENT_DIM)).astype(np.float32)
#
#    gen_inputs = {
#        latent_input_name: noise,
#        label_input_name: label_oh,
#    }
#    gen_images = gen_session.run([gen_output_name], gen_inputs)[0][:, :, :, 0]
#
#    preds = cls_session.run(
#        [cls_output_name],
#        {cls_input_name: gen_images.reshape(-1, 28, 28, 1).astype(np.float32)}
#    )[0]
#    preds = preds.argmax(axis=1)
#    accuracy = (preds == c).mean()
#    print(f"  Class {c}: {accuracy*100:.1f}% correctly classified ({int((preds == c).sum())}/{SAMPLES_PER_CLASS})")

# ─── Real data classification ────────────────────────────────────────
print(f"\n─── Real data classification (from synthetic project) ───")

for c in range(NUM_CLASSES):
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

print("\nDone!")
