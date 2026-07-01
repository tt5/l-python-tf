#!/usr/bin/env python3
"""train_cvae2.py

Train a Conditional Variational Autoencoder (CVAE) on 22x22 binary PBM images
from data/pbm/, upscaled to 28x28, with 11 classes (digits 0-9 + low_conf).

Usage:
    /home/n/miniconda3/envs/tf/bin/python train_cvae2.py

Saves:
    - cvae2_encoder.onnx
    - cvae2_generator.onnx
"""

import argparse
import tensorflow as tf
import numpy as np
import os
from pathlib import Path

DEFAULT_PBM_DIR = Path(__file__).resolve().parent / "../../telephonegame/data/pbm"

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=DEFAULT_PBM_DIR, help="PBM data directory")
args, _ = parser.parse_known_args()
PBM_DIR = args.data_dir

# ─── Hyperparameters ──────────────────────────────────────────────
LATENT_DIM = 64
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.002
NUM_CLASSES = None  # derived from data + low_conf
TARGET_SIZE = 28
LOW_CONF_THRESHOLD = 0.6
RECON_LOSS_WEIGHT = 1.0
KL_WEIGHT = 0.5
KL_WARMUP_EPOCHS = 2
KL_DECAY_EPOCHS = EPOCHS // 2
KL_WEIGHT_END = 0.1
LR_WARMUP_EPOCHS = 5
LR_START = 0.0
LR_DECAY_EPOCHS = EPOCHS // 2
LR_END = 0.0001
L2_REG = 0.00001
KL_FREE_BITS = 0.05
DROPOUT = 0.2

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
        img_28 = img_28.astype(np.float32)

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

    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(image_input)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    # Residual conv block
    x_res = x
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Add()([x, x_res])
    x = tf.keras.layers.Activation("relu")(x)
    # SE block
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Dense(64 // 4, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(se)
    se = tf.keras.layers.Dense(64, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(se)
    x = tf.keras.layers.Multiply()([x, se])
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Concatenate()([x, label_input])
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)

    z_mean = tf.keras.layers.Dense(LATENT_DIM, name="z_mean", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    z_log_var = tf.keras.layers.Dense(LATENT_DIM, name="z_log_var", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    z = Sampling()([z_mean, z_log_var])

    return tf.keras.Model([image_input, label_input], [z_mean, z_log_var, z], name="encoder")

# ─── Generator (Decoder) ─────────────────────────────────────────
def build_generator():
    latent_input = tf.keras.layers.Input(shape=(LATENT_DIM,), name="latent_input")
    label_input = tf.keras.layers.Input(shape=(NUM_CLASSES,), name="label_input")

    x = tf.keras.layers.Concatenate()([latent_input, label_input])
    x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)

    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
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
        self.kl_weight = 0.0

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
            kl_per_dim = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_per_dim = tf.maximum(kl_per_dim, KL_FREE_BITS)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_per_dim, axis=1))
            total_loss = (RECON_LOSS_WEIGHT * recon_loss) + (float(self.kl_weight) * kl_loss)

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
        kl_per_dim = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_per_dim = tf.maximum(kl_per_dim, KL_FREE_BITS)
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_per_dim, axis=1))

        total_loss = (RECON_LOSS_WEIGHT * recon_loss) + (float(self.kl_weight) * kl_loss)

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

    def generate(self, label, noise=None):
        """Generate an image from a class label (0-10)."""
        label_oh = tf.keras.utils.to_categorical([label], NUM_CLASSES)
        if noise is None:
            noise = tf.random.normal(shape=(1, LATENT_DIM))
        return self.generator([noise, label_oh])


class KLWarmup(tf.keras.callbacks.Callback):
    def __init__(self, cvae, warmup_epochs, kl_weight_final, decay_epochs, kl_weight_end):
        self.cvae = cvae
        self.warmup_epochs = warmup_epochs
        self.kl_weight_final = kl_weight_final
        self.decay_epochs = decay_epochs
        self.kl_weight_end = kl_weight_end

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            weight = self.kl_weight_final * (epoch / max(1, self.warmup_epochs - 1))
        elif epoch < self.warmup_epochs + self.decay_epochs:
            t = (epoch - self.warmup_epochs) / max(1, self.decay_epochs - 1)
            weight = self.kl_weight_final + (self.kl_weight_end - self.kl_weight_final) * t
        else:
            weight = self.kl_weight_end
        self.cvae.kl_weight = float(weight)


class LRWarmup(tf.keras.callbacks.Callback):
    def __init__(self, lr_start, lr_peak, warmup_epochs, lr_end, decay_epochs):
        self.lr_start = lr_start
        self.lr_peak = lr_peak
        self.warmup_epochs = warmup_epochs
        self.lr_end = lr_end
        self.decay_epochs = decay_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.lr_start + (self.lr_peak - self.lr_start) * (epoch / max(1, self.warmup_epochs - 1))
        elif epoch < self.warmup_epochs + self.decay_epochs:
            t = (epoch - self.warmup_epochs) / max(1, self.decay_epochs - 1)
            lr = self.lr_peak + (self.lr_end - self.lr_peak) * t
        else:
            lr = self.lr_end
        if self.model is not None and self.model.optimizer is not None:
            tf.keras.backend.set_value(self.model.optimizer.lr, float(lr))

# ─── Build & train ───────────────────────────────────────────────
def main():
    x_train, y_train, x_test, y_test = load_dataset()

    encoder = build_encoder()
    generator = build_generator()
    cvae = CVAE(encoder, generator)
    cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0))

    print("Encoder summary:")
    encoder.summary()
    print("\nGenerator summary:")
    generator.summary()

    print(f"\nTraining CVAE for {EPOCHS} epochs...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_total_loss',
        patience=8,
        restore_best_weights=True
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(Path(__file__).parent / "logs" / "train_cvae2"),
        histogram_freq=1,
        write_graph=True,
        update_freq="epoch",
    )
    kl_warmup = KLWarmup(cvae, warmup_epochs=KL_WARMUP_EPOCHS, kl_weight_final=KL_WEIGHT, decay_epochs=KL_DECAY_EPOCHS, kl_weight_end=KL_WEIGHT_END)
    lr_warmup = LRWarmup(lr_start=LR_START, lr_peak=LEARNING_RATE, warmup_epochs=LR_WARMUP_EPOCHS, lr_end=LR_END, decay_epochs=LR_DECAY_EPOCHS)
    cvae.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, tensorboard_callback, kl_warmup, lr_warmup],
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
    gen_path = save_dir / "cvae2_generator.onnx"
    onnx.save(onnx_model, str(gen_path))
    print(f"\nGenerator saved to {gen_path}")

    # Encoder: (image, label) -> z_mean, z_log_var, z
    enc_spec = (
        tf.TensorSpec((None, 28, 28, 1), tf.float32, name="image_input"),
        tf.TensorSpec((None, NUM_CLASSES), tf.float32, name="label_input"),
    )
    onnx_model, _ = tf2onnx.convert.from_keras(encoder, input_signature=enc_spec)
    enc_path = save_dir / "cvae2_encoder.onnx"
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
