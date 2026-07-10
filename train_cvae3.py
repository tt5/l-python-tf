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
LATENT_DIM = 384
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.009
LR_START = 0.00001
LR_WARMUP_EPOCHS = 9
LR_DECAY_EPOCHS = EPOCHS - LR_WARMUP_EPOCHS
LR_END = 0.00015
CLIPNORM = 1.0
KL_START = 0.000001
KL_PEAK = 0.45
KL_END = 0.38
KL_WARMUP_EPOCHS = 5
KL_DECAY_EPOCHS = 120 - KL_WARMUP_EPOCHS
DROPOUT = 0.45
L2_REG = 0.00005
CONTRASTIVE_TEMPERATURE = 0.07
CONTRASTIVE_START = 0.00005
CONTRASTIVE_PEAK = 2
CONTRASTIVE_END = 0
CONTRASTIVE_WARMUP_EPOCHS = 7
CONTRASTIVE_DECAY_EPOCHS = 0
PERCEPTUAL_START = 0.0
PERCEPTUAL_PEAK = 4.0
PERCEPTUAL_END = 0.0
PERCEPTUAL_WARMUP_EPOCHS = 20
PERCEPTUAL_DECAY_EPOCHS = 120 - PERCEPTUAL_WARMUP_EPOCHS
NUM_CLASSES = None  # derived from data + low_conf
TARGET_SIZE = 28
LOW_CONF_THRESHOLD = 0.6
TB_LOG_DIR = Path(__file__).resolve().parent / "logs" / "cvae3b"

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
    print(f"\nBefore aug - Train: {x_train.shape}, Test: {x_test.shape}")

    # Precompute 8-way augmentations for training images only
    x_train_aug = _build_aug_stack(x_train)
    print(f"Augmented train stack: {x_train_aug.shape}")

    return x_train_aug, y_train, x_test, y_test


def _build_aug_stack(images):
    t = np.transpose(images, axes=(0, 2, 1, 3))
    stacks = [
        images,
        np.rot90(images, k=1, axes=(1, 2)),
        np.rot90(images, k=2, axes=(1, 2)),
        np.rot90(images, k=3, axes=(1, 2)),
        t,
        np.rot90(t, k=1, axes=(1, 2)),
        np.rot90(t, k=2, axes=(1, 2)),
        np.rot90(t, k=3, axes=(1, 2)),
    ]
    return np.stack(stacks, axis=1)


def _random_aug_sample(stack, label):
    k = tf.random.uniform((), minval=0, maxval=8, dtype=tf.int32)
    img = stack[k]
    return img, label


# ─── Sampling layer ───────────────────────────────────────────────
class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class LinearLRSchedule(tf.keras.callbacks.Callback):
    def __init__(self, lr_start, lr_peak, lr_end, warmup_epochs, decay_epochs):
        self.lr_start = lr_start
        self.lr_peak = lr_peak
        self.lr_end = lr_end
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.lr_start + (self.lr_peak - self.lr_start) * (epoch / max(1, self.warmup_epochs - 1))
        elif epoch < self.warmup_epochs + self.decay_epochs:
            t = (epoch - self.warmup_epochs) / max(1, self.decay_epochs - 1)
            lr = self.lr_peak + (self.lr_end - self.lr_peak) * t
        else:
            lr = self.lr_end
        tf.keras.backend.set_value(self.model.optimizer.lr, float(lr))

class LogLRCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        self._log_dir = str(log_dir)
        self._writer = None

    def on_train_begin(self, logs=None):
        if self._writer is None:
            self._writer = tf.summary.create_file_writer(self._log_dir)

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        with self._writer.as_default():
            tf.summary.scalar('learning_rate', lr, step=epoch)

class LinearKLSchedule(tf.keras.callbacks.Callback):
    def __init__(self, kl_start, kl_peak, kl_end, warmup_epochs, decay_epochs):
        self.kl_start = kl_start
        self.kl_peak = kl_peak
        self.kl_end = kl_end
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self._writer = None
        self._log_dir = None

    def set_model(self, model):
        super().set_model(model)

    def on_train_begin(self, logs=None):
        return

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            weight = self.kl_start + (self.kl_peak - self.kl_start) * (epoch / max(1, self.warmup_epochs - 1))
        elif epoch < self.warmup_epochs + self.decay_epochs:
            t = (epoch - self.warmup_epochs) / max(1, self.decay_epochs - 1)
            weight = self.kl_peak + (self.kl_end - self.kl_peak) * t
        else:
            weight = self.kl_end
        self.model.kl_weight.assign(float(weight))

class ContrastiveWeightSchedule(tf.keras.callbacks.Callback):
    def __init__(self, start, peak, end, warmup_epochs, decay_epochs):
        self.start = start
        self.peak = peak
        self.end = end
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            weight = self.start + (self.peak - self.start) * (epoch / max(1, self.warmup_epochs - 1))
        elif epoch < self.warmup_epochs + self.decay_epochs:
            t = (epoch - self.warmup_epochs) / max(1, self.decay_epochs - 1)
            weight = self.peak + (self.end - self.peak) * t
        else:
            weight = self.end
        self.model.contrastive_weight.assign(float(weight))

class PerceptualWeightSchedule(tf.keras.callbacks.Callback):
    def __init__(self, start, peak, end, warmup_epochs, decay_epochs):
        self.start = start
        self.peak = peak
        self.end = end
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            weight = self.start + (self.peak - self.start) * (epoch / max(1, self.warmup_epochs - 1))
        elif epoch < self.warmup_epochs + self.decay_epochs:
            t = (epoch - self.warmup_epochs) / max(1, self.decay_epochs - 1)
            weight = self.peak + (self.end - self.peak) * t
        else:
            weight = self.end
        self.model.perceptual_weight.assign(float(weight))

def _make_perceptual_backbone():
    from mnist3 import poly_loss
    try:
        from tensorflow_addons.optimizers import AdamW
    except Exception:
        from tensorflow.keras.optimizers import Adam
        class AdamW(Adam):
            def __init__(self, weight_decay=0.0, **kwargs):
                super().__init__(**kwargs)
                self.weight_decay = weight_decay
    custom_objects = {'poly_loss': poly_loss, 'AdamW': AdamW}
    model_path = Path(__file__).parent / "mnist3_model"
    backbone = tf.keras.models.load_model(str(model_path), custom_objects=custom_objects)
    backbone.trainable = False
    out = backbone.get_layer("dense_5").output
    return tf.keras.Model(backbone.input, out, name="mnist3_backbone")

# ─── Encoder ─────────────────────────────────────────────────────
def build_encoder():
    image_input = tf.keras.layers.Input(shape=(28, 28, 1), name="image_input")
    label_input = tf.keras.layers.Input(shape=(NUM_CLASSES,), name="label_input")

    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(image_input)
    x = tf.keras.layers.LayerNormalization(center=False, scale=False)(x)
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    skip1 = x  # 14x14, 32ch

    # Multi-head label embedding: one head per spatial scale
    # 14x14 head
    enc_label_emb_head_14 = tf.keras.layers.Dense(128, activation="relu")(label_input)
    enc_label_emb_head_14 = tf.keras.layers.Dense(128, activation="relu")(enc_label_emb_head_14)
    enc_label_emb_head_14 = tf.keras.layers.Dense(64, activation=None)(enc_label_emb_head_14)
    film_g_14 = tf.keras.layers.Dense(32, activation="sigmoid")(enc_label_emb_head_14)
    film_b_14 = tf.keras.layers.Dense(32, activation=None)(enc_label_emb_head_14)
    film_g_14 = tf.keras.layers.Reshape((1, 1, 32))(film_g_14)
    film_b_14 = tf.keras.layers.Reshape((1, 1, 32))(film_b_14)
    x = tf.keras.layers.Multiply()([x, film_g_14])
    x = tf.keras.layers.Add()([x, film_b_14])

    # Residual downsampling into 64ch / 7x7
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization(center=False, scale=False)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Add()([
        x,
        tf.keras.layers.Conv2D(64, 1, activation=None, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(skip1)
    ])

    # 7x7 head
    enc_label_emb_head_7 = tf.keras.layers.Dense(128, activation="relu")(label_input)
    enc_label_emb_head_7 = tf.keras.layers.Dense(128, activation="relu")(enc_label_emb_head_7)
    enc_label_emb_head_7 = tf.keras.layers.Dense(64, activation=None)(enc_label_emb_head_7)
    film_g = tf.keras.layers.Dense(64, activation="sigmoid")(enc_label_emb_head_7)
    film_b = tf.keras.layers.Dense(64, activation=None)(enc_label_emb_head_7)
    film_g = tf.keras.layers.Reshape((1, 1, 64))(film_g)
    film_b = tf.keras.layers.Reshape((1, 1, 64))(film_b)
    x = tf.keras.layers.Multiply()([x, film_g])
    x = tf.keras.layers.Add()([x, film_b])

    # Multi-scale kernel block at 7x7
    b1 = tf.keras.layers.Conv2D(16, 1, padding='same', activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    b3 = tf.keras.layers.Conv2D(16, 3, padding='same', activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    b5 = tf.keras.layers.Conv2D(16, 5, padding='same', activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.Concatenate()([b1, b3, b5])
    x = tf.keras.layers.Conv2D(64, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)

    # SE on 7x7 spatial features
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Dense(64 // 4, activation="relu")(se)
    se = tf.keras.layers.Dense(64, activation="sigmoid")(se)
    se = tf.keras.layers.Reshape((1, 1, 64))(se)
    x = tf.keras.layers.Multiply()([x, se])

    # 7x7 head for broadcast label map
    label_map_head = tf.keras.layers.Dense(128, activation="relu")(enc_label_emb_head_7)
    label_map = tf.keras.layers.Dense(7 * 7 * 64, activation=None)(label_map_head)
    label_map = tf.keras.layers.Reshape((7, 7, 64))(label_map)
    label_map = tf.keras.layers.LayerNormalization(center=False, scale=False)(label_map)
    x = tf.keras.layers.Add()([x, label_map])

    x = tf.keras.layers.Dropout(DROPOUT)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Concatenate()([x, label_input])
    x_save = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation=None)(x_save)
    x = tf.keras.layers.LayerNormalization(center=False, scale=False)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Add()([x, x_save])

    # Dense-stage FiLM before latent stats using 7x7 head
    enc_film_emb = tf.keras.layers.Dense(128, activation="relu")(enc_label_emb_head_7)
    enc_film_g = tf.keras.layers.Dense(256, activation="sigmoid")(enc_film_emb)
    enc_film_b = tf.keras.layers.Dense(256, activation=None)(enc_film_emb)
    x = tf.keras.layers.Multiply()([x, enc_film_g])
    x = tf.keras.layers.Add()([x, enc_film_b])

    # deeper label injection before latent stats using 7x7 head
    enc_film_emb_deep = tf.keras.layers.Dense(128, activation="relu")(enc_label_emb_head_7)
    enc_film_g_deep = tf.keras.layers.Dense(256, activation="sigmoid")(enc_film_emb_deep)
    enc_film_b_deep = tf.keras.layers.Dense(256, activation=None)(enc_film_emb_deep)
    x = tf.keras.layers.Multiply()([x, enc_film_g_deep])
    x = tf.keras.layers.Add()([x, enc_film_b_deep])

    x = tf.keras.layers.Dropout(DROPOUT)(x)

    z_mean = tf.keras.layers.Dense(LATENT_DIM, name="z_mean", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    z_log_var = tf.keras.layers.Dense(LATENT_DIM, name="z_log_var", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    z = Sampling()([z_mean, z_log_var])
    z = tf.keras.layers.Dense(LATENT_DIM, activation="relu")(z)
    z = tf.keras.layers.LayerNormalization()(z)

    return tf.keras.Model([image_input, label_input], [z_mean, z_log_var, z], name="encoder")

# ─── Generator (Decoder) ─────────────────────────────────────────
def build_generator():
    latent_input = tf.keras.layers.Input(shape=(LATENT_DIM,), name="latent_input")
    label_input = tf.keras.layers.Input(shape=(NUM_CLASSES,), name="label_input")

    x = tf.keras.layers.Concatenate()([latent_input, label_input])
    x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization(center=False, scale=False)(x)

    # Second label injection: concat label again after latent LayerNorm, before reshape
    x_label = tf.keras.layers.Dense(64, activation="relu")(label_input)
    x_label = tf.keras.layers.Dense(7 * 7 * 64, activation=None)(x_label)
    x = tf.keras.layers.Concatenate()([x, x_label])
    x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)

    x = tf.keras.layers.Reshape((7, 7, 64))(x)

    # Multi-head label embedding: one head per spatial scale
    # 7x7 head
    dec_label_emb_head_7 = tf.keras.layers.Dense(128, activation="relu")(label_input)
    dec_label_emb_head_7 = tf.keras.layers.Dense(128, activation="relu")(dec_label_emb_head_7)
    dec_label_emb_head_7 = tf.keras.layers.Dense(64, activation=None)(dec_label_emb_head_7)
    film_g = tf.keras.layers.Dense(64, activation="sigmoid")(dec_label_emb_head_7)
    film_b = tf.keras.layers.Dense(64, activation=None)(dec_label_emb_head_7)
    film_g = tf.keras.layers.Reshape((1, 1, 64))(film_g)
    film_b = tf.keras.layers.Reshape((1, 1, 64))(film_b)
    x = tf.keras.layers.Multiply()([x, film_g])
    x = tf.keras.layers.Add()([x, film_b])

    # SE on 7x7 generator features
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Dense(64 // 4, activation="relu")(se)
    se = tf.keras.layers.Dense(64, activation="sigmoid")(se)
    se = tf.keras.layers.Reshape((1, 1, 64))(se)
    x = tf.keras.layers.Multiply()([x, se])

    # 7x7 -> 14x14 residual upsample block
    skip7 = x
    x = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization(center=False, scale=False)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Add()([
        x,
        tf.keras.layers.Conv2D(64, 1, padding='same')(
            tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')(skip7)
        )
    ])
    x = tf.keras.layers.Dropout(DROPOUT / 2)(x)

    # Multi-scale kernel block at 14x14
    b1 = tf.keras.layers.Conv2D(16, 1, padding='same', activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    b3 = tf.keras.layers.Conv2D(16, 3, padding='same', activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    b5 = tf.keras.layers.Conv2D(16, 5, padding='same', activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.Concatenate()([b1, b3, b5])
    x = tf.keras.layers.Conv2D(64, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)

    # 14x14 head
    dec_label_emb_head_14 = tf.keras.layers.Dense(128, activation="relu")(label_input)
    dec_label_emb_head_14 = tf.keras.layers.Dense(128, activation="relu")(dec_label_emb_head_14)
    dec_label_emb_head_14 = tf.keras.layers.Dense(64, activation=None)(dec_label_emb_head_14)
    film_g_14 = tf.keras.layers.Dense(64, activation="sigmoid")(dec_label_emb_head_14)
    film_b_14 = tf.keras.layers.Dense(64, activation=None)(dec_label_emb_head_14)
    film_g_14 = tf.keras.layers.Reshape((1, 1, 64))(film_g_14)
    film_b_14 = tf.keras.layers.Reshape((1, 1, 64))(film_b_14)
    x = tf.keras.layers.Multiply()([x, film_g_14])
    x = tf.keras.layers.Add()([x, film_b_14])

    # SE on 14x14 generator features
    se_14 = tf.keras.layers.GlobalAveragePooling2D()(x)
    se_14 = tf.keras.layers.Dense(64 // 4, activation="relu")(se_14)
    se_14 = tf.keras.layers.Dense(64, activation="sigmoid")(se_14)
    se_14 = tf.keras.layers.Reshape((1, 1, 64))(se_14)
    x = tf.keras.layers.Multiply()([x, se_14])

    # 14x14 -> 28x28 residual upsample block
    skip14 = x
    x = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization(center=False, scale=False)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Add()([
        x,
        tf.keras.layers.Conv2D(32, 1, padding='same')(
            tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')(skip14)
        )
    ])
    x = tf.keras.layers.Dropout(DROPOUT / 2)(x)

    # second 28x28 refinement block
    x28 = tf.keras.layers.Conv2D(32, 3, padding='same', activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x28 = tf.keras.layers.LayerNormalization(center=False, scale=False)(x28)

    # 28x28 head
    dec_label_emb_head_28 = tf.keras.layers.Dense(64, activation="relu")(label_input)
    dec_label_emb_head_28 = tf.keras.layers.Dense(64, activation="relu")(dec_label_emb_head_28)
    dec_label_emb_head_28 = tf.keras.layers.Dense(32, activation=None)(dec_label_emb_head_28)
    film_g_28 = tf.keras.layers.Dense(32, activation="sigmoid")(dec_label_emb_head_28)
    film_b_28 = tf.keras.layers.Dense(32, activation=None)(dec_label_emb_head_28)
    film_g_28 = tf.keras.layers.Reshape((1, 1, 32))(film_g_28)
    film_b_28 = tf.keras.layers.Reshape((1, 1, 32))(film_b_28)
    x28 = tf.keras.layers.Multiply()([x28, film_g_28])
    x28 = tf.keras.layers.Add()([x28, film_b_28])

    x = tf.keras.layers.Add()([x, x28])
    x = tf.keras.layers.Dropout(DROPOUT / 2)(x)

    x = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)

    return tf.keras.Model([latent_input, label_input], x, name="generator")

# ─── CVAE ────────────────────────────────────────────────────────
class CVAE(tf.keras.Model):
    def __init__(self, encoder, generator, perceptual_backbone=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.generator = generator
        self.perceptual_backbone = perceptual_backbone
        self.kl_weight = tf.Variable(KL_START, dtype=tf.float32, trainable=False)
        self.contrastive_weight = tf.Variable(CONTRASTIVE_START, dtype=tf.float32, trainable=False)
        self.perceptual_weight = tf.Variable(float(PERCEPTUAL_START), dtype=tf.float32, trainable=False)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.kl_weight_tracker = tf.keras.metrics.Mean(name="kl_weight")
        self.contrastive_loss_tracker = tf.keras.metrics.Mean(name="contrastive_loss")
        self.contrastive_weight_tracker = tf.keras.metrics.Mean(name="contrastive_weight")
        self.perceptual_loss_tracker = tf.keras.metrics.Mean(name="perceptual_loss")
        self.perceptual_weight_tracker = tf.keras.metrics.Mean(name="perceptual_weight")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker, self.kl_weight_tracker, self.contrastive_loss_tracker, self.contrastive_weight_tracker, self.perceptual_loss_tracker, self.perceptual_weight_tracker]

    def _contrastive_loss(self, z, labels):
        z_norm = tf.math.l2_normalize(z, axis=-1)
        logits = tf.matmul(z_norm, z_norm, transpose_b=True) / float(CONTRASTIVE_TEMPERATURE)
        labels_int = tf.cast(tf.argmax(labels, axis=1), tf.int32)
        same_mask = tf.cast(tf.equal(labels_int[:, None], labels_int[None, :]), tf.float32)
        same_mask = tf.linalg.set_diag(same_mask, tf.zeros_like(tf.linalg.diag_part(same_mask)))

        exp_logits = tf.exp(logits)
        row_sum = tf.reduce_sum(exp_logits, axis=1, keepdims=True) - tf.linalg.diag_part(exp_logits)[:, None]
        pos_sum = tf.reduce_sum(exp_logits * same_mask, axis=1, keepdims=True)
        has_pos = tf.reduce_any(same_mask > 0.0, axis=1, keepdims=True)
        denom = row_sum + pos_sum
        row_loss = -tf.math.log(tf.maximum(pos_sum, 1e-12)) + tf.math.log(tf.maximum(denom, 1e-12))
        row_loss = tf.where(tf.squeeze(has_pos, axis=1), tf.squeeze(row_loss, axis=1), tf.zeros_like(tf.squeeze(row_loss, axis=1)))
        return tf.reduce_mean(row_loss)

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
            contrastive_loss = self._contrastive_loss(z, labels)

            perceptual_loss = tf.constant(0.0)
            if self.perceptual_backbone is not None:
                logits_fake = self.perceptual_backbone(reconstruction)
                perceptual_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, logits_fake, from_logits=True))

            total_loss = recon_loss + self.kl_weight * kl_loss + self.contrastive_weight * contrastive_loss + self.perceptual_weight * perceptual_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_weight_tracker.update_state(self.kl_weight)
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        self.contrastive_weight_tracker.update_state(self.contrastive_weight)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.perceptual_weight_tracker.update_state(self.perceptual_weight)

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
        contrastive_loss = self._contrastive_loss(z, labels)

        perceptual_loss = tf.constant(0.0)
        if self.perceptual_backbone is not None:
            logits_fake = self.perceptual_backbone(reconstruction)
            perceptual_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, logits_fake, from_logits=True))

        total_loss = recon_loss + self.kl_weight * kl_loss + self.contrastive_weight * contrastive_loss + self.perceptual_weight * perceptual_loss

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_weight_tracker.update_state(self.kl_weight)
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        self.contrastive_weight_tracker.update_state(self.contrastive_weight)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.perceptual_weight_tracker.update_state(self.perceptual_weight)

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
    perceptual_backbone = _make_perceptual_backbone()
    cvae = CVAE(encoder, generator, perceptual_backbone=perceptual_backbone)
    cvae.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE, clipnorm=CLIPNORM))

    print("Encoder summary:")
    encoder.summary()
    print("\nGenerator summary:")
    generator.summary()

    print(f"\nTraining CVAE for {EPOCHS} epochs...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_recon_loss',
        patience=200,
        restore_best_weights=True
    )
    checkpoint_path = Path(__file__).parent / "best_recon.weights"
    checkpoint_path.unlink(missing_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_recon_loss',
        save_best_only=True,
        save_weights_only=True,
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(TB_LOG_DIR),
        update_freq='epoch'
    )
    lr_schedule = LinearLRSchedule(
        lr_start=LR_START,
        lr_peak=LEARNING_RATE,
        lr_end=LR_END,
        warmup_epochs=LR_WARMUP_EPOCHS,
        decay_epochs=LR_DECAY_EPOCHS
    )
    kl_schedule = LinearKLSchedule(
        kl_start=KL_START,
        kl_peak=KL_PEAK,
        kl_end=KL_END,
        warmup_epochs=KL_WARMUP_EPOCHS,
        decay_epochs=KL_DECAY_EPOCHS
    )
    contrastive_schedule = ContrastiveWeightSchedule(
        start=CONTRASTIVE_START,
        peak=CONTRASTIVE_PEAK,
        end=CONTRASTIVE_END,
        warmup_epochs=CONTRASTIVE_WARMUP_EPOCHS,
        decay_epochs=CONTRASTIVE_DECAY_EPOCHS
    )

    perceptual_schedule = PerceptualWeightSchedule(
        start=PERCEPTUAL_START,
        peak=PERCEPTUAL_PEAK,
        end=PERCEPTUAL_END,
        warmup_epochs=PERCEPTUAL_WARMUP_EPOCHS,
        decay_epochs=PERCEPTUAL_DECAY_EPOCHS,
    )

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(x_train), seed=42)
    train_ds = train_ds.map(_random_aug_sample, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    cvae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stopping, checkpoint, tensorboard_callback, lr_schedule, kl_schedule, contrastive_schedule, perceptual_schedule, LogLRCallback(TB_LOG_DIR)],
    )

    # ─── Save as ONNX from best restored weights ───────────────────
    save_dir = Path(__file__).parent

    import tf2onnx
    import onnx

    # Generator: (latent, label) -> image
    gen_spec = (
        tf.TensorSpec((None, LATENT_DIM), tf.float32, name="latent_input"),
        tf.TensorSpec((None, NUM_CLASSES), tf.float32, name="label_input"),
    )
    onnx_model, _ = tf2onnx.convert.from_keras(cvae.generator, input_signature=gen_spec)
    gen_path = save_dir / "cvae3_generator.onnx"
    onnx.save(onnx_model, str(gen_path))
    print(f"\nGenerator saved to {gen_path}")

    # Encoder: (image, label) -> z_mean, z_log_var, z
    enc_spec = (
        tf.TensorSpec((None, 28, 28, 1), tf.float32, name="image_input"),
        tf.TensorSpec((None, NUM_CLASSES), tf.float32, name="label_input"),
    )
    onnx_model, _ = tf2onnx.convert.from_keras(cvae.encoder, input_signature=enc_spec)
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
