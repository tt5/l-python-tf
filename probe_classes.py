#!/usr/bin/env python3
"""probe_classes.py

Quick probe to estimate per-class difficulty.
Trains a lightweight model for PROBE_EPOCHS with standard cross-entropy,
then computes per-class validation accuracy and derives focal-loss alpha weights.

Usage:
    python probe_classes.py
    python probe_classes.py --num-classes 5
    python probe_classes.py --epochs 15
"""

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# ── Defaults (CLI can override) ─────────────────────────────────────
PROBE_EPOCHS = 10
BATCH_SIZE = 128
DENSE_SIZE = 256          # smaller than main model (512)
DROPOUT = 0.30
L2_DECAY = 1e-5
LEARNING_RATE = 0.005
VAL_SPLIT = 0.5

# ── Argparse ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num-classes", type=int, default=5)
parser.add_argument("--epochs", type=int, default=PROBE_EPOCHS)
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
args = parser.parse_args()
NUM_CLASSES = args.num_classes
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

# ── Data ─────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic" / "data"

def load_synthetic():
    train_images, train_labels = [], []
    test_images, test_labels = [], []

    for class_id in range(NUM_CLASSES):
        class_dir = DATA_DIR / f"class_{class_id}"
        files = sorted(class_dir.glob("*.npy"))
        split_idx = int(len(files) * VAL_SPLIT)
        for f in files[:split_idx]:
            img = np.load(str(f)).astype(np.float32) / 255.0
            train_images.append(img)
            train_labels.append(class_id)
        for f in files[split_idx:]:
            img = np.load(str(f)).astype(np.float32) / 255.0
            test_images.append(img)
            test_labels.append(class_id)

    train_x = np.array(train_images).reshape(-1, 28, 28, 1)
    train_y = np.array(train_labels, dtype=np.int64)
    test_x = np.array(test_images).reshape(-1, 28, 28, 1)
    test_y = np.array(test_labels, dtype=np.int64)

    perm = np.random.permutation(len(train_x))
    train_x, train_y = train_x[perm], train_y[perm]
    perm = np.random.permutation(len(test_x))
    test_x, test_y = test_x[perm], test_y[perm]

    return (train_x, train_y), (test_x, test_y)


(x_train, y_train), (x_test, y_test) = load_synthetic()
print(f"Train: {len(x_train)}, Test: {len(x_test)}")

# ── Lightweight probe model ──────────────────────────────────────────
L = tf.keras.layers
inputs = L.Input(shape=(28, 28, 1))
x = inputs

# Single CBAM-like block (simpler: no separable depthwise, plain Conv2D)
x = L.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
x = L.MaxPooling2D(2)(x)
x = L.Dropout(DROPOUT)(x)

# Channel attention (SE)
channels = x.shape[-1]
gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
gmp = tf.reduce_max(x, axis=[1, 2], keepdims=True)
mlp = tf.keras.Sequential([
    L.Dense(channels // 8, activation='relu'),
    L.Dense(channels, activation='sigmoid')
])
x = tf.multiply(x, tf.add(mlp(gap), mlp(gmp)))

# Spatial attention
avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
concat = tf.concat([avg_pool, max_pool], axis=-1)
x = tf.multiply(x, L.Conv2D(1, 7, padding='same', activation='sigmoid')(concat))

# Second conv block
x = L.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
x = L.MaxPooling2D(2)(x)
x = L.Dropout(DROPOUT)(x)

# Flatten + dense
x = L.Flatten()(x)
x = L.Dense(DENSE_SIZE, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
x = L.Dropout(DROPOUT)(x)
x = L.Dense(DENSE_SIZE, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
x = L.Dropout(DROPOUT)(x)

outputs = L.Dense(NUM_CLASSES)(x)
model = tf.keras.Model(inputs, outputs)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=loss_fn,
    metrics=['accuracy']
)

# ── Train ────────────────────────────────────────────────────────────
model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0
)

# ── Per-class accuracy ──────────────────────────────────────────────
y_pred_logits = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_logits, axis=1)

per_class_acc = {}
for c in range(NUM_CLASSES):
    mask = (y_test == c)
    if np.sum(mask) == 0:
        per_class_acc[c] = 0.0
    else:
        per_class_acc[c] = np.mean(y_pred[mask] == y_test[mask])

print("\n─── Per-class validation accuracy ───")
for c in range(NUM_CLASSES):
    print(f"  Class {c}: {per_class_acc[c]*100:.1f}%")
avg_acc = np.mean([per_class_acc[c] for c in range(NUM_CLASSES)])
print(f"  Average: {avg_acc*100:.1f}%")

# ── Derive focal alpha weights ──────────────────────────────────────
# alpha_i ∝ 1 / acc_i  (inverse difficulty)
accs = np.array([per_class_acc[c] for c in range(NUM_CLASSES)], dtype=np.float32)
# avoid division by zero
accs = np.clip(accs, 1e-6, 1.0)
raw_alpha = 1.0 / accs
alpha = raw_alpha / np.sum(raw_alpha)

print("\n─── Suggested focal alpha weights ───")
for c in range(NUM_CLASSES):
    print(f"  Class {c}: {alpha[c]:.4f}")
print(f"  Sum: {np.sum(alpha):.4f}")
print("\nPaste these into mnist.py as:")
print(f"  FOCAL_ALPHA = {list(np.round(alpha, 4))}")
