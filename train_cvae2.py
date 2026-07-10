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
LATENT_DIM = 256
EPOCHS = 180
BATCH_SIZE = 64
NUM_CLASSES = None  # derived from data + low_conf
TARGET_SIZE = 28
LOW_CONF_THRESHOLD = 0.6
KL_WARMUP_EPOCHS = 2
KL_START_WEIGHT = 0.09
KL_WEIGHT = 0.45
KL_DECAY_EPOCHS = EPOCHS // 2
KL_WEIGHT_END = 0.04
LEARNING_RATE = 0.0021
LR_WARMUP_EPOCHS = 8
LR_START = 0.000087
LR_DECAY_EPOCHS = EPOCHS - LR_WARMUP_EPOCHS
LR_END = 0.000015
L2_REG = 0.00001
KL_FREE_BITS = 0.0
DROPOUT = 0.37
AUGMENT = True
RECON_LOSS_WEIGHT = 0.35
PERCEPTUAL_FINE_WEIGHT = 1.0
PERCEPTUAL_COARSE_WEIGHT = 1.0
PERCEPTUAL_MULTISCALE = True
PERCEPTUAL_WEIGHTS = [PERCEPTUAL_FINE_WEIGHT, PERCEPTUAL_COARSE_WEIGHT]
PERCEPTUAL_WARMUP_EPOCHS = 9
PERCEPTUAL_START_WEIGHT = 0.006
PERCEPTUAL_PEAK_WEIGHT = 65
PERCEPTUAL_DECAY_EPOCHS = 60
PERCEPTUAL_END_WEIGHT = 0
PERCEPTUAL_EXTRACTOR_PATH = Path(__file__).parent / "mnist2_model"
TV_WARMUP_EPOCHS = 2
TV_START_WEIGHT = 0.07
TV_PEAK_WEIGHT = 0.17
TV_DECAY_EPOCHS = 50
TV_END_WEIGHT = 0.0

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
def label_cross_attention(spatial, label_input, channels, l2_reg):
    B = tf.shape(spatial)[0]
    H = int(spatial.shape[1])
    W = int(spatial.shape[2])
    flat = H * W

    le = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(label_input)
    le = tf.keras.layers.Dense(64, activation=None, kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(le)

    q = tf.keras.layers.Dense(channels)(spatial)
    q = tf.reshape(q, [B, flat, channels])

    k = tf.keras.layers.Dense(channels)(le)
    v = tf.keras.layers.Dense(channels)(le)

    scores = tf.matmul(q, tf.expand_dims(k, -1)) / tf.sqrt(tf.cast(channels, tf.float32))
    attn = tf.nn.softmax(scores, axis=1)

    ctx = attn * tf.expand_dims(v, 1)
    ctx = tf.reshape(ctx, [B, H, W, channels])
    ctx = tf.keras.layers.LayerNormalization(center=False, scale=False)(ctx)

    return tf.keras.layers.Add()([spatial, ctx])

def build_encoder():
    image_input = tf.keras.layers.Input(shape=(28, 28, 1), name="image_input")
    label_input = tf.keras.layers.Input(shape=(NUM_CLASSES,), name="label_input")

    x = tf.keras.layers.Conv2D(32, 3, strides=1, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(image_input)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.SpatialDropout2D(DROPOUT)(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x14 = x  # 14x14, 32 channels

    # SE block at 14x14
    se14_enc = tf.keras.layers.GlobalAveragePooling2D()(x14)
    se14_enc = tf.keras.layers.Dense(32 // 4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(se14_enc)
    se14_enc = tf.keras.layers.Dense(32, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(se14_enc)
    x14 = tf.keras.layers.Multiply()([x14, se14_enc])

    # Label pathway
    label_emb = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(label_input)

    # Cross-attention at 14x14
    x14 = label_cross_attention(x14, label_input, 32, L2_REG)

    # 14x14 residual block
    x14_res = x14
    x14 = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x14)
    x14 = tf.keras.layers.LayerNormalization()(x14)
    x14 = tf.keras.layers.Conv2D(32, 3, padding="same", activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x14)
    x14 = tf.keras.layers.LayerNormalization()(x14)
    x14 = tf.keras.layers.Add()([x14, x14_res])
    x14 = tf.keras.layers.Activation("relu")(x14)

    x = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x14)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.SpatialDropout2D(DROPOUT)(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    # Residual conv block
    x_res = x
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Add()([x, x_res])
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.SpatialDropout2D(DROPOUT)(x)
    # SE block
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Dense(64 // 4, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(se)
    se = tf.keras.layers.Dense(64, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(se)
    x = tf.keras.layers.Multiply()([x, se])
    x = tf.keras.layers.SpatialDropout2D(DROPOUT)(x)
    # Multi-scale kernel block at 7x7
    b1 = tf.keras.layers.Conv2D(32, 1, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    b3 = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    b5 = tf.keras.layers.Conv2D(32, 5, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.Concatenate()([b1, b3, b5])
    x = tf.keras.layers.Conv2D(64, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.SpatialDropout2D(DROPOUT)(x)
    # Label pathway
    label_emb = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(label_input)

    x = label_cross_attention(x, label_input, 64, L2_REG)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Concatenate()([x, label_emb])
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    z_mean = tf.keras.layers.Dense(LATENT_DIM, name="z_mean", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    z_log_var = tf.keras.layers.Dense(LATENT_DIM, name="z_log_var", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    z = Sampling()([z_mean, z_log_var])

    return tf.keras.Model([image_input, label_input], [z_mean, z_log_var, z], name="encoder")

# ─── Generator (Decoder) ─────────────────────────────────────────
def build_generator():
    latent_input = tf.keras.layers.Input(shape=(LATENT_DIM,), name="latent_input")
    label_input = tf.keras.layers.Input(shape=(NUM_CLASSES,), name="label_input")

    # Label pathway
    label_emb = tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(label_input)
    label_emb = tf.keras.layers.Dense(96, activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(label_emb)

    # Concatenate latent and label for initial expansion
    z = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(latent_input)
    x = tf.keras.layers.Concatenate()([z, label_emb])
    x = tf.keras.layers.Dense(7 * 7 * 96, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.Reshape((7, 7, 96))(x)
    x = label_cross_attention(x, label_input, 96, L2_REG)
    x = tf.keras.layers.SpatialDropout2D(DROPOUT)(x)
    skip7 = x

    # SE block at 7x7
    se7 = tf.keras.layers.GlobalAveragePooling2D()(skip7)
    se7 = tf.keras.layers.Dense(96 // 4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(se7)
    se7 = tf.keras.layers.Dense(96, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(se7)
    skip7 = tf.keras.layers.Multiply()([skip7, se7])
    x = tf.keras.layers.SpatialDropout2D(DROPOUT)(x)
    x = tf.keras.layers.Conv2DTranspose(48, 2, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.Conv2D(48, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization(center=False, scale=False)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = label_cross_attention(x, label_input, 48, L2_REG)
    x = tf.keras.layers.Add()([
        x,
        tf.keras.layers.Conv2DTranspose(48, 2, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(
            skip7
        )
    ])

    # 14x14 block
    x = label_cross_attention(x, label_input, 48, L2_REG)

    # Multi-scale kernel block at 14x14
    b1 = tf.keras.layers.Conv2D(16, 1, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    b3 = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    b5 = tf.keras.layers.Conv2D(16, 5, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.Concatenate()([b1, b3, b5])
    x = tf.keras.layers.Conv2D(48, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.SpatialDropout2D(DROPOUT)(x)

    skip14 = x
    # SE block at 14x14
    se14 = tf.keras.layers.GlobalAveragePooling2D()(skip14)
    se14 = tf.keras.layers.Dense(48 // 4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(se14)
    se14 = tf.keras.layers.Dense(48, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(se14)
    skip14 = tf.keras.layers.Multiply()([skip14, se14])
    x = tf.keras.layers.SpatialDropout2D(DROPOUT)(x)
    x = tf.keras.layers.Conv2DTranspose(48, 2, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.Conv2D(48, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization(center=False, scale=False)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = label_cross_attention(x, label_input, 48, L2_REG)
    x = tf.keras.layers.Add()([
        x,
        tf.keras.layers.Conv2DTranspose(48, 2, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(
            skip14
        )
    ])
    x = tf.keras.layers.SpatialDropout2D(DROPOUT)(x)

    # 28x28 block
    x = label_cross_attention(x, label_input, 48, L2_REG)

    x = tf.keras.layers.SpatialDropout2D(DROPOUT)(x)
    x = tf.keras.layers.Conv2D(48, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)

    # SE block at 28x28
    se28 = tf.keras.layers.GlobalAveragePooling2D()(x)
    se28 = tf.keras.layers.Dense(48 // 4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(se28)
    se28 = tf.keras.layers.Dense(48, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(se28)
    x = tf.keras.layers.Multiply()([x, se28])

    x = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    return tf.keras.Model([latent_input, label_input], x, name="generator")

# ─── CVAE ────────────────────────────────────────────────────────
class CVAE(tf.keras.Model):
    def __init__(self, encoder, generator, perceptual_extractor=None, perceptual_coarse_extractor=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.generator = generator
        self.perceptual_extractor = perceptual_extractor
        self.perceptual_coarse_extractor = perceptual_coarse_extractor
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.perceptual_loss_tracker = tf.keras.metrics.Mean(name="perceptual_loss")
        self.perceptual_fine_loss_tracker = tf.keras.metrics.Mean(name="perceptual_fine_loss")
        self.perceptual_coarse_loss_tracker = tf.keras.metrics.Mean(name="perceptual_coarse_loss")
        self.kl_weight = 0.0
        self.perceptual_weight = float(PERCEPTUAL_PEAK_WEIGHT)
        self.perceptual_coarse_weight = float(PERCEPTUAL_COARSE_WEIGHT)
        self.tv_weight = float(TV_PEAK_WEIGHT)

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker, self.perceptual_loss_tracker, self.perceptual_fine_loss_tracker, self.perceptual_coarse_loss_tracker]

    def train_step(self, data):
        images, labels = data
        if AUGMENT:
            candidates = tf.concat([
                images,
                tf.image.rot90(images, k=1),
                tf.image.rot90(images, k=2),
                tf.image.rot90(images, k=3),
                tf.image.transpose(images),
            ], axis=0)
            n = tf.shape(images)[0]
            idx = tf.random.uniform((n,), minval=0, maxval=5, dtype=tf.int32)
            images = tf.gather(candidates, idx)
            labels = tf.gather(labels, idx)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([images, labels])
            reconstruction = self.generator([z, labels])

            recon_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(images, reconstruction),
                    axis=(1, 2)
                )
            )
            tv_loss = tf.reduce_mean(
                tf.image.total_variation(reconstruction)
            )
            kl_per_dim = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_per_dim = tf.maximum(kl_per_dim, KL_FREE_BITS)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_per_dim, axis=1))

            perceptual_loss_fine = 0.0
            if self.perceptual_extractor is not None:
                real_input = tf.squeeze(images, -1)
                fake_input = tf.squeeze(reconstruction, -1)
                real_feats = self.perceptual_extractor(real_input, training=False)
                fake_feats = self.perceptual_extractor(fake_input, training=False)
                perceptual_loss_fine = tf.reduce_mean(tf.reduce_sum(tf.square(real_feats - fake_feats), axis=-1))

            perceptual_loss_coarse = 0.0
            if self.perceptual_coarse_extractor is not None:
                real_input = tf.squeeze(images, -1)
                fake_input = tf.squeeze(reconstruction, -1)
                real_coarse = self.perceptual_coarse_extractor(real_input, training=False)
                fake_coarse = self.perceptual_coarse_extractor(fake_input, training=False)
                perceptual_loss_coarse = tf.reduce_mean(tf.reduce_sum(tf.square(real_coarse - fake_coarse), axis=-1))

            if PERCEPTUAL_MULTISCALE:
                perceptual_loss = PERCEPTUAL_WEIGHTS[0] * perceptual_loss_fine + PERCEPTUAL_WEIGHTS[1] * perceptual_loss_coarse
            else:
                perceptual_loss = perceptual_loss_fine + perceptual_loss_coarse

            total_loss = (RECON_LOSS_WEIGHT * recon_loss) + (float(self.perceptual_weight) * perceptual_loss) + (float(self.kl_weight) * kl_loss) + float(self.tv_weight) * tv_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.perceptual_fine_loss_tracker.update_state(perceptual_loss_fine)
        self.perceptual_coarse_loss_tracker.update_state(perceptual_loss_coarse)

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

        perceptual_loss = 0.0
        perceptual_coarse_loss = 0.0
        if self.perceptual_extractor is not None:
            real_input = tf.squeeze(images, -1)
            fake_input = tf.squeeze(reconstruction, -1)
            real_feats = self.perceptual_extractor(real_input, training=False)
            fake_feats = self.perceptual_extractor(fake_input, training=False)
            perceptual_loss_fine = tf.reduce_mean(tf.reduce_sum(tf.square(real_feats - fake_feats), axis=-1))
        else:
            perceptual_loss_fine = tf.constant(0.0)

        if self.perceptual_coarse_extractor is not None:
            real_input = tf.squeeze(images, -1)
            fake_input = tf.squeeze(reconstruction, -1)
            real_coarse = self.perceptual_coarse_extractor(real_input, training=False)
            fake_coarse = self.perceptual_coarse_extractor(fake_input, training=False)
            perceptual_loss_coarse = tf.reduce_mean(tf.reduce_sum(tf.square(real_coarse - fake_coarse), axis=-1))
        else:
            perceptual_loss_coarse = tf.constant(0.0)

        if PERCEPTUAL_MULTISCALE:
            perceptual_loss = PERCEPTUAL_WEIGHTS[0] * perceptual_loss_fine + PERCEPTUAL_WEIGHTS[1] * perceptual_loss_coarse
        else:
            perceptual_loss = perceptual_loss_fine + perceptual_loss_coarse

        total_loss = (RECON_LOSS_WEIGHT * recon_loss) + (float(self.perceptual_weight) * perceptual_loss) + (float(self.kl_weight) * kl_loss)

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.perceptual_fine_loss_tracker.update_state(perceptual_loss_fine)
        self.perceptual_coarse_loss_tracker.update_state(perceptual_loss_coarse)

        return {m.name: m.result() for m in self.metrics}

    def generate(self, label, noise=None):
        """Generate an image from a class label (0-10)."""
        label_oh = tf.keras.utils.to_categorical([label], NUM_CLASSES)
        if noise is None:
            noise = tf.random.normal(shape=(1, LATENT_DIM))
        return self.generator([noise, label_oh])


def build_perceptual_extractor():
    """Load frozen mnist2 classifier and build explicit fine/coarse conv sub-models by name."""
    base = tf.keras.models.load_model(str(PERCEPTUAL_EXTRACTOR_PATH), compile=False)
    base.trainable = False

    fine_layer = base.get_layer(name='conv14').output
    coarse_layer = base.get_layer(name='conv7_deep2').output

    gap_fine = tf.keras.layers.GlobalAveragePooling2D()(fine_layer)
    gap_coarse = tf.keras.layers.GlobalAveragePooling2D()(coarse_layer)

    fine_extractor = tf.keras.Model(inputs=base.input, outputs=gap_fine, name='perceptual_extractor_14x14')
    coarse_extractor = tf.keras.Model(inputs=base.input, outputs=gap_coarse, name='perceptual_extractor_7x7')
    return fine_extractor, coarse_extractor


class KLWarmup(tf.keras.callbacks.Callback):
    def __init__(self, cvae, warmup_epochs, start_weight, kl_weight_final, decay_epochs, kl_weight_end):
        self.cvae = cvae
        self.warmup_epochs = warmup_epochs
        self.start_weight = start_weight
        self.kl_weight_final = kl_weight_final
        self.decay_epochs = decay_epochs
        self.kl_weight_end = kl_weight_end

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            weight = self.start_weight + (self.kl_weight_final - self.start_weight) * (epoch / max(1, self.warmup_epochs - 1))
        elif epoch < self.warmup_epochs + self.decay_epochs:
            t = (epoch - self.warmup_epochs) / max(1, self.decay_epochs - 1)
            weight = self.kl_weight_final + (self.kl_weight_end - self.kl_weight_final) * t
        else:
            weight = self.kl_weight_end
        self.cvae.kl_weight = float(weight)


class PerceptualWarmup(tf.keras.callbacks.Callback):
    def __init__(self, cvae, warmup_epochs, start_weight, peak_weight, decay_epochs, end_weight):
        self.cvae = cvae
        self.warmup_epochs = warmup_epochs
        self.start_weight = start_weight
        self.peak_weight = peak_weight
        self.decay_epochs = decay_epochs
        self.end_weight = end_weight

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            weight = self.start_weight + (self.peak_weight - self.start_weight) * (epoch / max(1, self.warmup_epochs - 1))
        elif epoch < self.warmup_epochs + self.decay_epochs:
            t = (epoch - self.warmup_epochs) / max(1, self.decay_epochs - 1)
            weight = self.peak_weight + (self.end_weight - self.peak_weight) * t
        else:
            weight = self.end_weight
        self.cvae.perceptual_weight = float(weight)


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


class TVWarmup(tf.keras.callbacks.Callback):
    def __init__(self, cvae, warmup_epochs, start_weight, peak_weight, decay_epochs, end_weight):
        self.cvae = cvae
        self.warmup_epochs = warmup_epochs
        self.start_weight = start_weight
        self.peak_weight = peak_weight
        self.decay_epochs = decay_epochs
        self.end_weight = end_weight

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            weight = self.start_weight + (self.peak_weight - self.start_weight) * (epoch / max(1, self.warmup_epochs - 1))
        elif epoch < self.warmup_epochs + self.decay_epochs:
            t = (epoch - self.warmup_epochs) / max(1, self.decay_epochs - 1)
            weight = self.peak_weight + (self.end_weight - self.peak_weight) * t
        else:
            weight = self.end_weight
        self.cvae.tv_weight = float(weight)

# ─── Build & train ───────────────────────────────────────────────
def main():
    x_train, y_train, x_test, y_test = load_dataset()

    encoder = build_encoder()
    generator = build_generator()
    perceptual_extractor, perceptual_coarse_extractor = build_perceptual_extractor()
    cvae = CVAE(encoder, generator, perceptual_extractor=perceptual_extractor, perceptual_coarse_extractor=perceptual_coarse_extractor)
    cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0))

    print("Encoder summary:")
    encoder.summary()
    print("\nGenerator summary:")
    generator.summary()

    print(f"\nTraining CVAE for {EPOCHS} epochs...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_total_loss',
        patience=15,
        restore_best_weights=True
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(Path(__file__).parent / "logs" / "train_cvae1"),
        histogram_freq=1,
        write_graph=True,
        update_freq="epoch",
    )
    kl_warmup = KLWarmup(cvae, warmup_epochs=KL_WARMUP_EPOCHS, start_weight=KL_START_WEIGHT, kl_weight_final=KL_WEIGHT, decay_epochs=KL_DECAY_EPOCHS, kl_weight_end=KL_WEIGHT_END)
    lr_warmup = LRWarmup(lr_start=LR_START, lr_peak=LEARNING_RATE, warmup_epochs=LR_WARMUP_EPOCHS, lr_end=LR_END, decay_epochs=LR_DECAY_EPOCHS)
    perceptual_warmup = PerceptualWarmup(cvae, warmup_epochs=PERCEPTUAL_WARMUP_EPOCHS, start_weight=PERCEPTUAL_START_WEIGHT, peak_weight=PERCEPTUAL_PEAK_WEIGHT, decay_epochs=PERCEPTUAL_DECAY_EPOCHS, end_weight=PERCEPTUAL_END_WEIGHT)
    tv_warmup = TVWarmup(cvae, warmup_epochs=TV_WARMUP_EPOCHS, start_weight=TV_START_WEIGHT, peak_weight=TV_PEAK_WEIGHT, decay_epochs=TV_DECAY_EPOCHS, end_weight=TV_END_WEIGHT)
    cvae.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, tensorboard_callback, kl_warmup, lr_warmup, perceptual_warmup, tv_warmup],
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
