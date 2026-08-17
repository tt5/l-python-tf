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
EPOCHS = 110
BATCH_SIZE = 128
FREEZE_EPOCHS = 1
NUM_CLASSES = None  # derived from data + low_conf
TARGET_SIZE = 28
LOW_CONF_THRESHOLD = 0.6
KL_WARMUP_EPOCHS = 5
KL_START_WEIGHT = 0.02
KL_WEIGHT = 0.45
KL_DECAY_EPOCHS = EPOCHS - KL_WARMUP_EPOCHS
KL_WEIGHT_END = 0.1
LEARNING_RATE = 0.0005
LR_WARMUP_EPOCHS = 4
LR_START = 0.0002
LR_DECAY_EPOCHS = EPOCHS - LR_WARMUP_EPOCHS
LR_END = 0.00001
L2_REG = 0.000006
KL_FREE_BITS = 0.0
SPATIAL_DROPOUT = 0.35
DENSE_DROPOUT = 0.35
GEN_SPATIAL_DROPOUT = 0.3
GEN_DENSE_DROPOUT = 0.3
AUGMENT = True
RECON_LOSS_WEIGHT = 1.0
PERCEPTUAL_MULTISCALE = True
PERCEPTUAL_WARMUP_EPOCHS = 5
PERCEPTUAL_START_WEIGHT = 0.0
PERCEPTUAL_PEAK_WEIGHT = 1
PERCEPTUAL_DECAY_EPOCHS = EPOCHS - PERCEPTUAL_WARMUP_EPOCHS
PERCEPTUAL_END_WEIGHT = 0.5
PERCEPTUAL_EXTRACTOR_PATH = Path(__file__).parent / "mnist2_model"
TV_WARMUP_EPOCHS = 5
TV_START_WEIGHT = 0.2
TV_PEAK_WEIGHT = 0.2
TV_DECAY_EPOCHS = EPOCHS - TV_WARMUP_EPOCHS
TV_END_WEIGHT = 0.2
LOG_NAME = "train_gen2b"

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
        # Binarize (should already be binary from upscale, but ensure)
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
def label_cross_attention(spatial, label_input, channels, l2_reg):
    B = tf.shape(spatial)[0]
    H = int(spatial.shape[1])
    W = int(spatial.shape[2])
    flat = H * W

    le = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(label_input)
    le = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(le)

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

def make_film(channels, emb=None):
    def film_fn(label):
        if emb is not None:
            shared_emb = emb
        else:
            emb_dim = max(16, channels // 4)
            shared_emb = tf.keras.layers.Dense(emb_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(label)
        gamma = tf.keras.layers.Dense(channels, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(shared_emb)
        beta = tf.keras.layers.Dense(channels, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(shared_emb)
        gamma = tf.keras.layers.Reshape((1, 1, channels))(gamma)
        beta = tf.keras.layers.Reshape((1, 1, channels))(beta)
        return gamma, beta
    return film_fn


def channel_attention(x, reduction=16, name_prefix='cbam'):
    channels = x.shape[-1]
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    gmp = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    mlp = tf.keras.Sequential([
        tf.keras.layers.Dense(channels // reduction, activation='relu', name=f'{name_prefix}_ca_dense1'),
        tf.keras.layers.Dense(channels, activation='sigmoid', name=f'{name_prefix}_ca_dense2')
    ])
    attn = tf.add(mlp(gap), mlp(gmp))
    return tf.multiply(x, attn)


def spatial_attention(x, name_prefix='cbam'):
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    attn = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', name=f'{name_prefix}_sa_conv1')(concat)
    attn = tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid', name=f'{name_prefix}_sa_conv2')(attn)
    return tf.multiply(x, attn)


def cbam_block(x, name_prefix='cbam'):
    x = channel_attention(x, name_prefix=name_prefix)
    x = spatial_attention(x, name_prefix=name_prefix)
    x = tf.keras.layers.LayerNormalization(name=f'{name_prefix}_ln')(x)
    return x


def build_encoder():
    image_input = tf.keras.layers.Input(shape=(28, 28, 1), name="image_input")
    label_input = tf.keras.layers.Input(shape=(NUM_CLASSES,), name="label_input")

    # Alternative path: transferred mnist2 layers + CBAM
    x28_alt = image_input
    x28_alt_res = x28_alt
    x28_alt = tf.keras.layers.SeparableConv2D(32, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='conv28')(x28_alt)
    x28_alt = tf.keras.layers.SeparableConv2D(32, 3, padding='same', activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='conv28_second')(x28_alt)
    x28_alt = tf.keras.layers.LayerNormalization(name='enc_ln28_2')(x28_alt)
    x28_alt = tf.keras.layers.Add()([x28_alt, x28_alt_res])
    x28_alt = tf.keras.layers.Activation('relu')(x28_alt)
    x28_alt = tf.keras.layers.Add()([x28_alt, cbam_block(x28_alt, name_prefix='cbam28')])


    # ── 14x14 block ──────────────────────────────────────────────
    x28 = x28_alt
    x28_block_out = x28
    x14_main = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_main14_conv')(x28)
    x14_main = tf.keras.layers.MaxPooling2D(2)(x14_main)
    x14_main = tf.keras.layers.LayerNormalization(name='enc_main14_ln')(x14_main)

    # Two residual conv blocks in main path
    x14_main_res = x14_main
    x14_main = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_main14_res1')(x14_main)
    x14_main = tf.keras.layers.Add()([x14_main, x14_main_res])
    x14_main = tf.keras.layers.LayerNormalization(name='enc_main14_res1_ln')(x14_main)
    x14_main = tf.keras.layers.Activation('relu')(x14_main)
    x14_main = tf.keras.layers.SpatialDropout2D(SPATIAL_DROPOUT)(x14_main)

    x14_main_res2 = x14_main
    x14_main = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_main14_res2')(x14_main)
    x14_main = tf.keras.layers.Add()([x14_main, x14_main_res2])
    x14_main = tf.keras.layers.LayerNormalization(name='enc_main14_res2_ln')(x14_main)
    x14_main = tf.keras.layers.Activation('relu')(x14_main)
    x14_main = tf.keras.layers.SpatialDropout2D(SPATIAL_DROPOUT)(x14_main)

    # SE block on main path
    se14 = tf.keras.layers.GlobalAveragePooling2D()(x14_main)
    se14 = tf.keras.layers.Dense(64 // 4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_se14_dense1')(se14)
    se14 = tf.keras.layers.Dense(64, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_se14_dense2')(se14)
    x14_main = tf.keras.layers.Multiply()([x14_main, se14])

    x14_alt = tf.keras.layers.SeparableConv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='conv14_pre')(x28)
    x14_alt = tf.keras.layers.MaxPooling2D(2)(x14_alt)
    x14_alt_res = x14_alt
    x14_alt = tf.keras.layers.SeparableConv2D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='conv14_res1')(x14_alt)
    x14_alt = tf.keras.layers.Add()([x14_alt, x14_alt_res])
    x14_alt = tf.keras.layers.Activation('relu')(x14_alt)
    x14_alt_res2 = x14_alt
    x14_alt = tf.keras.layers.SeparableConv2D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='conv14_res2')(x14_alt)
    x14_alt = tf.keras.layers.LayerNormalization(name='enc_ln14_2')(x14_alt)
    x14_alt = tf.keras.layers.Add()([x14_alt, x14_alt_res2])
    x14_alt = tf.keras.layers.Activation('relu')(x14_alt)
    x14_alt = tf.keras.layers.Add()([x14_alt, cbam_block(x14_alt, name_prefix='cbam14')])
    x14 = x14_alt

    gate14 = tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc14_skip_gate')(x28_block_out)
    gate14 = tf.keras.layers.MaxPooling2D(2)(gate14)
    x = gate14 * x14_main + (1.0 - gate14) * x14_alt

    x14 = tf.keras.layers.LayerNormalization(name='enc_ln14_pre')(x)
    x14_pre = x14
    film_gamma14, film_beta14 = make_film(64)(label_input)
    x14 = x14 * (1.0 + film_gamma14) + film_beta14
    x14 = tf.keras.layers.LayerNormalization(name='enc_ln14_film')(x14)
    x14 = tf.keras.layers.Add()([
        x14,
        tf.keras.layers.Conv2D(64, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_skip14_1x1')(x14_pre),
    ])

    # ── 7x7 block ───────────────────────────────────────────────
    x14_block_out = x14
    x7_main = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_main7_conv')(x14)
    x7_main = tf.keras.layers.MaxPooling2D(2)(x7_main)
    x7_main = tf.keras.layers.LayerNormalization(name='enc_main7_ln')(x7_main)

    x7_alt = tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='conv7')(x14)
    x7_alt = tf.keras.layers.MaxPooling2D(2)(x7_alt)
    x7_alt = tf.keras.layers.Add()([x7_alt, cbam_block(x7_alt, name_prefix='cbam7')])
    x7_alt_shallow = x7_alt
    x7_alt = tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='conv7_deep2')(x7_alt)
    x7_alt = tf.keras.layers.LayerNormalization(name='enc_ln7_1')(x7_alt)
    x7_alt_res = x7_alt
    x7_alt = tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='conv7_res1')(x7_alt)
    x7_alt = tf.keras.layers.Add()([x7_alt, x7_alt_res])
    x7_alt = tf.keras.layers.Activation('relu', name='enc_act7')(x7_alt)
    x7_alt = tf.keras.layers.Add(name='enc_add7')([x7_alt_shallow, x7_alt])
    x7_alt = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_fuse7')(x7_alt)
    x7_alt = tf.keras.layers.LayerNormalization(name='enc_fuse7_ln')(x7_alt)

    # ── Latent head: progressive fusion ──────────────────────────
    # x: 7x7x128, x14: 14x14x64, x28: 28x28x32
    gate7 = tf.keras.layers.GlobalAveragePooling2D()(x14_block_out)
    gate7 = tf.keras.layers.Dense(128, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc7_skip_gate')(gate7)
    gate7 = tf.keras.layers.Reshape((1, 1, 128))(gate7)
    x7 = gate7 * x7_main + (1.0 - gate7) * x7_alt
    x7_gate_out = x7

    film_gamma7, film_beta7 = make_film(128)(label_input)
    x7 = x7 * (1.0 + film_gamma7) + film_beta7
    x7 = tf.keras.layers.LayerNormalization(name='enc_ln7_film')(x7)

    # 7x7 -> 14x14
    x7 = tf.keras.layers.UpSampling2D(2, name='enc_up7_14')(x7)
    x14_proj = tf.keras.layers.Conv2D(128, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_proj14')(x14)
    x7 = tf.keras.layers.Add()([x7, x14_proj])
    x7 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_fuse14')(x7)
    x7 = tf.keras.layers.LayerNormalization(name='enc_fuse14_ln')(x7)

    # 14x14 -> 28x28
    x7 = tf.keras.layers.UpSampling2D(2, name='enc_up14_28')(x7)
    x28_proj = tf.keras.layers.Conv2D(128, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_proj28')(x28)
    x7 = tf.keras.layers.Add()([x7, x28_proj])
    x7 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_fuse28')(x7)
    x7 = tf.keras.layers.LayerNormalization(name='enc_fuse28_ln')(x7)

    # Pool to vector
    x = tf.keras.layers.GlobalAveragePooling2D(name='enc_gap_fused')(x7)
    x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_main_proj')(x)
    x = tf.keras.layers.LayerNormalization(name='enc_main_ln')(x)
    x = tf.keras.layers.Dropout(DENSE_DROPOUT, name='enc_main_drop')(x)

    # Direct skip from original 7x7 features
    x_direct = tf.keras.layers.Flatten(name='enc_flat_direct')(x7_gate_out)
    x_direct = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_direct_proj1')(x_direct)
    x_direct = tf.keras.layers.LayerNormalization(name='enc_direct_ln1')(x_direct)
    x_direct = tf.keras.layers.Dropout(DENSE_DROPOUT, name='enc_direct_drop')(x_direct)
    x_direct = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='enc_direct_proj')(x_direct)
    x_direct = tf.keras.layers.LayerNormalization(name='enc_direct_ln2')(x_direct)
    x_direct = tf.keras.layers.Dropout(DENSE_DROPOUT, name='enc_direct_drop2')(x_direct)


    x = tf.keras.layers.Concatenate(name="enc_concat")([x, x_direct])
    x = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_fuse_proj")(x)
    x = tf.keras.layers.LayerNormalization(name="enc_ln512_1")(x)
    x = tf.keras.layers.Dropout(DENSE_DROPOUT, name="enc_drop512")(x)

    z_mean = tf.keras.layers.Dense(LATENT_DIM, name="z_mean", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    z_log_var = tf.keras.layers.Dense(LATENT_DIM, name="z_log_var", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    z = Sampling()([z_mean, z_log_var])

    return tf.keras.Model([image_input, label_input], [z_mean, z_log_var, z], name="encoder")

# ─── Generator (Decoder) ─────────────────────────────────────────
def build_generator():
    latent_input = tf.keras.layers.Input(shape=(LATENT_DIM,), name="latent_input")
    label_input = tf.keras.layers.Input(shape=(NUM_CLASSES,), name="label_input")

    # Class conditioning via direct label input at each scale
    label_emb = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_label_emb")(label_input)
    label_emb = tf.keras.layers.Dropout(GEN_DENSE_DROPOUT, name="gen_label_emb_drop")(label_emb)

    # Direct expansion: latent -> 7x7 spatial features, no discard
    x = tf.keras.layers.Dense(7 * 7 * 128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(latent_input)
    x = tf.keras.layers.Reshape((7, 7, 128))(x)
    x = tf.keras.layers.LayerNormalization()(x)

    # 7x7 stem: stabilize expanded features before residual block
    x = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Class conditioning at 7x7
    x7_pre = x
    x = label_cross_attention(x, label_input, 128, L2_REG)
    x = tf.keras.layers.Add()([
        x,
        tf.keras.layers.Conv2D(128, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x7_pre),
    ])

    # ── 7x7 block ───────────────────────────────────────────────
    skip7 = x

    # 7x7 block: 128 channels
    x7_res = x
    x = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Add()([x, x7_res])
    x = tf.keras.layers.Activation("relu")(x)

    # SE block at 7x7
    se7 = tf.keras.layers.GlobalAveragePooling2D()(x)
    se7 = tf.keras.layers.Dense(128 // 4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(se7)
    se7 = tf.keras.layers.Dense(128, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(se7)
    x = tf.keras.layers.Multiply()([x, se7])

    # ── 14x14 block ──────────────────────────────────────────────
    x = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.SpatialDropout2D(GEN_SPATIAL_DROPOUT)(x)
    skip7_up = tf.keras.layers.Conv2DTranspose(64, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(skip7)
    skip7_up = tf.keras.layers.Conv2D(64, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(skip7_up)

    # Spatial learned gating on skip path from shared label conditioning
    skip7_gate = tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(skip7_up)
    skip7_up = skip7_up * skip7_gate
    skip7_up = tf.keras.layers.LayerNormalization(center=True, scale=True)(skip7_up)

    x = tf.keras.layers.Add()([x, skip7_up])
    x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)

    # 14x14 block: 128 -> 64 channels
    channels14 = 64
    x14_pre = x
    x = label_cross_attention(x, label_input, channels14, L2_REG)
    scale, offset = make_film(channels14, emb=label_emb)(label_input)
    scale = tf.keras.layers.Reshape((1, 1, channels14))(scale)
    offset = tf.keras.layers.Reshape((1, 1, channels14))(offset)
    x = x * (1.0 + scale) + offset

    # Conv after FiLM
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)

    # Skip from pre-Film path so the model can weight raw features
    x = tf.keras.layers.Add()([
        x,
        tf.keras.layers.Conv2D(64, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x14_pre),
    ])

    skip14 = x
    x = tf.keras.layers.SpatialDropout2D(GEN_SPATIAL_DROPOUT)(x)

    # ── 28x28 block ──────────────────────────────────────────────
    x = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.SpatialDropout2D(GEN_SPATIAL_DROPOUT)(x)
    skip14_up = tf.keras.layers.Conv2DTranspose(32, 4, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(skip14)
    skip14_up = tf.keras.layers.Conv2D(32, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(skip14_up)

    skip14_gate = tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(skip14_up)
    skip14_up = skip14_up * skip14_gate
    skip14_up = tf.keras.layers.LayerNormalization(center=True, scale=True)(skip14_up)

    x = tf.keras.layers.Add()([x, skip14_up])
    x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)

    # 28x28 refinement: 32 channels
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)

    channels28 = 32
    x28_pre = x
    x = label_cross_attention(x28_pre, label_input, channels28, L2_REG)
    scale, offset = make_film(channels28, emb=label_emb)(label_input)
    scale = tf.keras.layers.Reshape((1, 1, channels28))(scale)
    offset = tf.keras.layers.Reshape((1, 1, channels28))(offset)
    x = x * (1.0 + scale) + offset

    # Conv after FiLM
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)

    # Skip from pre-Film path so the model can weight raw features
    x = tf.keras.layers.Add()([
        x,
        tf.keras.layers.Conv2D(32, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x28_pre),
    ])
    x = tf.keras.layers.SpatialDropout2D(GEN_SPATIAL_DROPOUT)(x)

    # post-FiLM refinement at 28x28 before the final 1x1
    x28_refine = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x28_refine = tf.keras.layers.LayerNormalization(center=True, scale=True)(x28_refine)
    x28_refine = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x28_refine)
    x28_refine = tf.keras.layers.LayerNormalization(center=True, scale=True)(x28_refine)
    x = tf.keras.layers.Add()([x, x28_refine])

    x = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    return tf.keras.Model([latent_input, label_input], x, name="generator")

# ─── CVAE ────────────────────────────────────────────────────────
class CVAE(tf.keras.Model):
    def __init__(self, encoder, generator, perceptual_extractor=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.generator = generator
        self.perceptual_extractor = perceptual_extractor
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.perceptual_loss_tracker = tf.keras.metrics.Mean(name="perceptual_loss")
        self.perceptual_conv28_loss_tracker = tf.keras.metrics.Mean(name="perceptual_conv28_loss")
        self.perceptual_conv14_loss_tracker = tf.keras.metrics.Mean(name="perceptual_conv14_loss")
        self.perceptual_conv7_loss_tracker = tf.keras.metrics.Mean(name="perceptual_conv7_loss")
        self.kl_weight = 0.0
        self.perceptual_weight = float(PERCEPTUAL_PEAK_WEIGHT)
        self.tv_weight = float(TV_PEAK_WEIGHT)
        self.perceptual_scale_weights = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG)),
            tf.keras.layers.Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="perceptual_scale_weights"),
        ], name="perceptual_scale_weights")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
            self.perceptual_loss_tracker,
            self.perceptual_conv28_loss_tracker,
            self.perceptual_conv14_loss_tracker,
            self.perceptual_conv7_loss_tracker,
        ]

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

            perceptual_loss_conv28 = 0.0
            perceptual_loss_conv14 = 0.0
            perceptual_loss_conv7 = 0.0
            if self.perceptual_extractor is not None:
                real_input = tf.squeeze(images, -1)
                fake_input = tf.squeeze(reconstruction, -1)
                real_feats = self.perceptual_extractor(real_input, training=False)
                fake_feats = self.perceptual_extractor(fake_input, training=False)
                perceptual_loss_conv28 = tf.reduce_mean(tf.square(real_feats[0] - fake_feats[0]))
                perceptual_loss_conv14 = tf.reduce_mean(tf.square(real_feats[1] - fake_feats[1]))
                perceptual_loss_conv7 = tf.reduce_mean(tf.square(real_feats[2] - fake_feats[2]))
                perceptual_loss_conv28 = perceptual_loss_conv28 / tf.cast(tf.shape(real_feats[0])[1] * tf.shape(real_feats[0])[2], tf.float32)
                perceptual_loss_conv14 = perceptual_loss_conv14 / tf.cast(tf.shape(real_feats[1])[1] * tf.shape(real_feats[1])[2], tf.float32)
                perceptual_loss_conv7 = perceptual_loss_conv7 / tf.cast(tf.shape(real_feats[2])[1] * tf.shape(real_feats[2])[2], tf.float32)

            if PERCEPTUAL_MULTISCALE:
                scale_losses = tf.stack([perceptual_loss_conv28, perceptual_loss_conv14, perceptual_loss_conv7])
                scale_losses = tf.expand_dims(scale_losses, 0)
                scale_weights = self.perceptual_scale_weights(scale_losses)
                perceptual_loss = tf.squeeze(tf.reduce_sum(scale_weights * scale_losses, axis=-1))
            else:
                perceptual_loss = perceptual_loss_conv28 + perceptual_loss_conv14 + perceptual_loss_conv7

            total_loss = (RECON_LOSS_WEIGHT * recon_loss) + (float(self.perceptual_weight) * perceptual_loss) + (float(self.kl_weight) * kl_loss) + float(self.tv_weight) * tv_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.perceptual_conv28_loss_tracker.update_state(perceptual_loss_conv28)
        self.perceptual_conv14_loss_tracker.update_state(perceptual_loss_conv14)
        self.perceptual_conv7_loss_tracker.update_state(perceptual_loss_conv7)

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
        tv_loss = tf.reduce_mean(
            tf.image.total_variation(reconstruction)
        )
        kl_per_dim = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_per_dim = tf.maximum(kl_per_dim, KL_FREE_BITS)
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_per_dim, axis=1))

        perceptual_loss_conv28 = 0.0
        perceptual_loss_conv14 = 0.0
        perceptual_loss_conv7 = 0.0
        if self.perceptual_extractor is not None:
            real_input = tf.squeeze(images, -1)
            fake_input = tf.squeeze(reconstruction, -1)
            real_feats = self.perceptual_extractor(real_input, training=False)
            fake_feats = self.perceptual_extractor(fake_input, training=False)
            perceptual_loss_conv28 = tf.reduce_mean(tf.square(real_feats[0] - fake_feats[0]))
            perceptual_loss_conv14 = tf.reduce_mean(tf.square(real_feats[1] - fake_feats[1]))
            perceptual_loss_conv7 = tf.reduce_mean(tf.square(real_feats[2] - fake_feats[2]))
            perceptual_loss_conv28 = perceptual_loss_conv28 / tf.cast(tf.shape(real_feats[0])[1] * tf.shape(real_feats[0])[2], tf.float32)
            perceptual_loss_conv14 = perceptual_loss_conv14 / tf.cast(tf.shape(real_feats[1])[1] * tf.shape(real_feats[1])[2], tf.float32)
            perceptual_loss_conv7 = perceptual_loss_conv7 / tf.cast(tf.shape(real_feats[2])[1] * tf.shape(real_feats[2])[2], tf.float32)
        else:
            perceptual_loss_conv28 = tf.constant(0.0)
            perceptual_loss_conv14 = tf.constant(0.0)
            perceptual_loss_conv7 = tf.constant(0.0)

        if PERCEPTUAL_MULTISCALE:
            scale_losses = tf.stack([perceptual_loss_conv28, perceptual_loss_conv14, perceptual_loss_conv7])
            scale_losses = tf.expand_dims(scale_losses, 0)
            scale_weights = self.perceptual_scale_weights(scale_losses)
            perceptual_loss = tf.squeeze(tf.reduce_sum(scale_weights * scale_losses, axis=-1))
        else:
            perceptual_loss = perceptual_loss_conv28 + perceptual_loss_conv14 + perceptual_loss_conv7

        total_loss = (RECON_LOSS_WEIGHT * recon_loss) + (float(self.perceptual_weight) * perceptual_loss) + (float(self.kl_weight) * kl_loss) + float(self.tv_weight) * tv_loss

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.perceptual_conv28_loss_tracker.update_state(perceptual_loss_conv28)
        self.perceptual_conv14_loss_tracker.update_state(perceptual_loss_conv14)
        self.perceptual_conv7_loss_tracker.update_state(perceptual_loss_conv7)

        return {m.name: m.result() for m in self.metrics}

    def generate(self, label, noise=None):
        """Generate an image from a class label (0-10)."""
        label_oh = tf.keras.utils.to_categorical([label], NUM_CLASSES)
        if noise is None:
            noise = tf.random.normal(shape=(1, LATENT_DIM))
        return self.generator([noise, label_oh])


def build_perceptual_extractor():
    """Load frozen mnist2 classifier and build a multi-scale feature extractor."""
    base = tf.keras.models.load_model(str(PERCEPTUAL_EXTRACTOR_PATH), compile=False)
    base.trainable = False

    conv28_layer = base.get_layer(name='conv28').output
    fine_layer = base.get_layer(name='conv14_pre').output
    coarse_layer = base.get_layer(name='conv7').output

    extractor = tf.keras.Model(inputs=base.input, outputs=[conv28_layer, fine_layer, coarse_layer], name='perceptual_extractor')
    return extractor


def copy_mnist2_weights_to_encoder(encoder):
    """Copy matching layer weights from mnist2_model into the encoder."""
    base = tf.keras.models.load_model(str(PERCEPTUAL_EXTRACTOR_PATH), compile=False)
    base.trainable = False

    copied = []
    skipped = []
    for layer in encoder.layers:
        if layer.name not in [l.name for l in base.layers]:
            skipped.append((layer.name, 'missing in mnist2_model'))
            continue
        base_layer = base.get_layer(name=layer.name)
        if layer.get_weights() and base_layer.get_weights():
            if all(s1 == s2 for s1, s2 in zip(layer.input_shape[1:], base_layer.input_shape[1:])):
                layer.set_weights(base_layer.get_weights())
                copied.append(layer.name)
            else:
                skipped.append((layer.name, f'shape mismatch: encoder={layer.input_shape} vs mnist2={base_layer.input_shape}'))
        else:
            skipped.append((layer.name, 'no weights'))

    print(f"\nWeight transfer from mnist2_model: {len(copied)} layers copied")
    for name in copied:
        print(f"  copied: {name}")
    if skipped:
        print(f"  skipped: {len(skipped)} layers")
        for name, reason in skipped:
            print(f"    {name}: {reason}")
    return copied


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
    copied_layer_names = copy_mnist2_weights_to_encoder(encoder)
    fusion_frozen = [
        ## progressive fusion
        #'enc_fuse7',
        #'enc_fuse7_ln',
        #'enc_proj14',
        #'enc_fuse14',
        #'enc_fuse14_ln',
        #'enc_proj28',
        #'enc_fuse28',
        #'enc_fuse28_ln',
        # alt path conv / res blocks
        'conv28',
        'conv28_second',
        'enc_ln28_2',
        'conv14_pre',
        'conv14_res1',
        'conv14_res2',
        'enc_ln14_2',
        'conv7',
        'conv7_deep2',
        'enc_ln7_1',
        'conv7_res1',
        # CBAM sublayers
        'cbam28_ca_dense1',
        'cbam28_ca_dense2',
        'cbam28_sa_conv1',
        'cbam28_sa_conv2',
        'cbam28_ln',
        'cbam14_ca_dense1',
        'cbam14_ca_dense2',
        'cbam14_sa_conv1',
        'cbam14_sa_conv2',
        'cbam14_ln',
        'cbam7_ca_dense1',
        'cbam7_ca_dense2',
        'cbam7_sa_conv1',
        'cbam7_sa_conv2',
        'cbam7_ln',
    ]
    frozen = [name for name in copied_layer_names if name in fusion_frozen]
    for name in frozen:
        encoder.get_layer(name=name).trainable = False
    print(f"Frozen transferred layers: {frozen}")
    generator = build_generator()
    perceptual_extractor = build_perceptual_extractor()
    cvae = CVAE(encoder, generator, perceptual_extractor=perceptual_extractor)
    cvae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0))

    print("Encoder summary:")
    encoder.summary()
    print("\nGenerator summary:")
    generator.summary()

    print(f"\nTraining CVAE for {EPOCHS} epochs...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_total_loss',
        patience=150,
        restore_best_weights=True
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(Path(__file__).parent / "logs" / LOG_NAME),
        histogram_freq=1,
        write_graph=True,
        update_freq="epoch",
    )
    kl_warmup = KLWarmup(cvae, warmup_epochs=KL_WARMUP_EPOCHS, start_weight=KL_START_WEIGHT, kl_weight_final=KL_WEIGHT, decay_epochs=KL_DECAY_EPOCHS, kl_weight_end=KL_WEIGHT_END)
    lr_warmup = LRWarmup(lr_start=LR_START, lr_peak=LEARNING_RATE, warmup_epochs=LR_WARMUP_EPOCHS, lr_end=LR_END, decay_epochs=LR_DECAY_EPOCHS)
    perceptual_warmup = PerceptualWarmup(cvae, warmup_epochs=PERCEPTUAL_WARMUP_EPOCHS, start_weight=PERCEPTUAL_START_WEIGHT, peak_weight=PERCEPTUAL_PEAK_WEIGHT, decay_epochs=PERCEPTUAL_DECAY_EPOCHS, end_weight=PERCEPTUAL_END_WEIGHT)
    tv_warmup = TVWarmup(cvae, warmup_epochs=TV_WARMUP_EPOCHS, start_weight=TV_START_WEIGHT, peak_weight=TV_PEAK_WEIGHT, decay_epochs=TV_DECAY_EPOCHS, end_weight=TV_END_WEIGHT)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

    print(f"\nTraining CVAE for {EPOCHS} epochs with {len(copied_layer_names)} transferred layers frozen...")
    cvae.compile(optimizer=optimizer)
    history = cvae.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, tensorboard_callback, kl_warmup, lr_warmup, perceptual_warmup, tv_warmup],
        verbose=2,
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
