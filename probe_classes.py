#!/usr/bin/env python3
"""probe_classes.py

Lightweight probe model for quick training runs.
Mirrors mnist.py architecture (CBAM + residual dense blocks).
Saves model to mnist_model/ for manual ONNX conversion and evaluation.

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
DENSE_SIZE = 512          # same as main model
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

# ── CBAM (same as mnist.py) ──────────────────────────────────────────
L = tf.keras.layers
# CBAM: Convolutional Block Attention Module
def channel_attention(x, reduction=16):
    channels = x.shape[-1]
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    gmp = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    mlp = tf.keras.Sequential([
        L.Dense(channels // reduction, activation='relu'),
        L.Dense(channels, activation='sigmoid')
    ])
    attn = tf.add(mlp(gap), mlp(gmp))
    return tf.multiply(x, attn)

def spatial_attention(x):
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    attn = L.Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
    return tf.multiply(x, attn)

def cbam_block(x):
    x = channel_attention(x)
    x = spatial_attention(x)
    return x

inputs = L.Input(shape=(28, 28, 1))
x = inputs

# CBAM attention blocks (same as mnist.py)
c1 = L.SeparableConv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
c1 = L.MaxPooling2D(2)(c1)
c1 = L.Dropout(DROPOUT)(c1)
c1 = cbam_block(c1)

# Conv block 2: 14x14 → 7x7
c2 = L.SeparableConv2D(128, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(c1)
c2 = L.MaxPooling2D(2)(c2)
c2 = L.Dropout(DROPOUT)(c2)
c2 = cbam_block(c2)

# Conv block 3: 7x7 → 7x7
c3 = L.SeparableConv2D(128, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(c2)
c3 = L.Dropout(DROPOUT)(c3)
c3 = cbam_block(c3)

# Flatten + Dense with residual connections
x = L.Flatten()(c3)
d1 = L.Dense(DENSE_SIZE, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
d1 = L.Dropout(DROPOUT)(d1)
p1 = L.Dense(DENSE_SIZE, kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
x = L.Add()([d1, p1])
x = L.Activation('relu')(x)

d2 = L.Dense(DENSE_SIZE, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
d2 = L.Dropout(DROPOUT)(d2)
p2 = L.Dense(DENSE_SIZE, kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
x = L.Add()([d2, p2])
x = L.Activation('relu')(x)

outputs = L.Dense(NUM_CLASSES)(x)
model = tf.keras.Model(inputs, outputs)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipvalue=1.0),
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

# ── Save ─────────────────────────────────────────────────────────────
out_dir = Path(__file__).resolve().parent / "mnist_model_probe"
model.save(out_dir)
print(f"\nModel saved to {out_dir}/")
