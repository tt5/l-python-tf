#!/usr/bin/env python3
"""train_cvae3.py

Train a Conditional Variational Autoencoder (CVAE) on 22x22 binary PBM images
from data/pbm2/, upscaled to 28x28, with 12 classes (digits 0-9 + low_conf + low_conf_2).

Usage:
    /home/n/miniconda3/envs/tf/bin/python train_cvae3.py

Saves:
    - cvae3_encoder.onnx
    - cvae3_generator.onnx
"""

import argparse
import tensorflow as tf
import numpy as np
import os
from pathlib import Path

DEFAULT_PBM_DIR = Path(__file__).resolve().parent / "../../telephonegame/data/pbm2"

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=DEFAULT_PBM_DIR, help="PBM data directory")
args, _ = parser.parse_known_args()
PBM_DIR = args.data_dir

# ─── Hyperparameters ──────────────────────────────────────────────
LATENT_DIM = 64
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_CLASSES = None  # derived from data + low_conf
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

def load_dataset():
    """Load all PBM images and labels."""
    files = sorted(PBM_DIR.glob("*.pbm"))
    print(f"Loading {len(files)} PBM images from {PBM_DIR}...")

    images = []
    labels = []

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
        else:
            labels.append(predicted_digit)

        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1}/{len(files)}")

    images = np.array(images)[..., np.newaxis]  # (N, 28, 28, 1)
    labels = np.array(labels)

    # Derive num_classes from data: max label + 1 for low_conf
    global NUM_CLASSES
    max_label = labels[labels >= 0].max()
    low_conf_label = max_label + 1
    labels[labels == -1] = low_conf_label
    NUM_CLASSES = low_conf_label + 1

    print(f"\nDerived NUM_CLASSES = {NUM_CLASSES} (max_label={max_label}, low_conf={low_conf_label})")

    # One-hot encode labels
    labels_oh = tf.keras.utils.to_categorical(labels, NUM_CLASSES)

    # Shuffle
    np.random.seed(42)
    perm = np.random.permutation(len(images))
    images = images[perm]
    labels_oh = labels_oh[perm]

    # Train/test split
    split = int(len(images) * 0.8)
    x_train, x_test = images[:split], images[split:]
    y_train, y_test = labels_oh[:split], labels_oh[split:]
    print(f"\nTrain: {len(x_train)}, Test: {len(x_test)}")

    return x_train, y_train, x_test, y_test

# ─── Sampling layer ───────────────────────────────────────────────
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ─── Encoder ─────────────────────────────────────────────────────
def build_encoder():
    image_input = tf.keras.layers.Input(shape=(28, 28, 1), name="image_input")
    label_input = tf.keras.layers.Input(shape=(NUM_CLASSES,), name="label_input")

    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(image_input)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Concatenate()([x, label_input])
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    z_mean = tf.keras.layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])

    return tf.keras.Model([image_input, label_input], [z_mean, z_log_var, z], name="encoder")

# ─── Generator (Decoder) ─────────────────────────────────────────
def build_generator():
    latent_input = tf.keras.layers.Input(shape=(LATENT_DIM,), name="latent_input")
    label_input = tf.keras.layers.Input(shape=(NUM_CLASSES,), name="label_input")

    x = tf.keras.layers.Concatenate()([latent_input, label_input])
    x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu")(x)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)

    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)

    return tf.keras.Model([latent_input, label_input], x, name="generator")

# ─── CVAE ────────────────────────────────────────────────────────
class CVAE(tf.keras.Model):
    def __init__(self, encoder, generator, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.generator = generator
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([images, labels])
            reconstruction = self.generator([z, labels])

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(images, reconstruction),
                    axis=(1, 2)
                )
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1
                )
            )
            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        z_mean, z_log_var, z = self.encoder([images, labels])
        reconstruction = self.generator([z, labels])

        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(images, reconstruction),
                axis=(1, 2)
            )
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )
        total_loss = recon_loss + kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def generate(self, label, noise=None):
        """Generate an image from a class label (0-11)."""
        label_oh = tf.keras.utils.to_categorical([label], NUM_CLASSES)
        if noise is None:
            noise = tf.random.normal(shape=(1, LATENT_DIM))
        return self.generator([noise, label_oh])

# ─── Build & train ───────────────────────────────────────────────
def main():
    x_train, y_train, x_test, y_test = load_dataset()

    encoder = build_encoder()
    generator = build_generator()
    cvae = CVAE(encoder, generator)
    cvae.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))

    print("Encoder summary:")
    encoder.summary()
    print("\nGenerator summary:")
    generator.summary()

    print(f"\nTraining CVAE for {EPOCHS} epochs...")
    cvae.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # ─── Save as ONNX ──────────────────────────────────────────────
    save_dir = Path(__file__).parent

    # Save generator as ONNX
    import tf2onnx
    import onnx

    # Generator: (latent, label) -> image
    gen_spec = (
        tf.TensorSpec((None, LATENT_DIM), tf.float32, name="latent_input"),
        tf.TensorSpec((None, NUM_CLASSES), tf.float32, name="label_input"),
    )
    onnx_model, _ = tf2onnx.convert.from_keras(generator, input_signature=gen_spec)
    gen_path = save_dir / "cvae3_generator.onnx"
    onnx.save(onnx_model, str(gen_path))
    print(f"\nGenerator saved to {gen_path}")

    # Encoder: (image, label) -> z_mean, z_log_var, z
    enc_spec = (
        tf.TensorSpec((None, 28, 28, 1), tf.float32, name="image_input"),
        tf.TensorSpec((None, NUM_CLASSES), tf.float32, name="label_input"),
    )
    onnx_model, _ = tf2onnx.convert.from_keras(encoder, input_signature=enc_spec)
    enc_path = save_dir / "cvae3_encoder.onnx"
    onnx.save(onnx_model, str(enc_path))
    print(f"Encoder saved to {enc_path}")

    # ─── Quick test: generate one sample per class ─────────────────
    print("\nGenerating test samples...")
    for label in range(NUM_CLASSES):
        img = cvae.generate(label)
        print(f"  Label {label}: shape={img.shape}, min={img.numpy().min():.3f}, max={img.numpy().max():.3f}")

    print("\nDone!")

if __name__ == "__main__":
    main()
