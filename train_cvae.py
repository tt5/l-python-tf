#!/usr/bin/env python3
"""train_cvae.py

Simple Conditional VAE for synthetic pictogram data.
Compatible with convert_generator.py for ONNX conversion.

Usage:
    python train_cvae.py
"""

import numpy as np
import os
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path

from collab import CollabModulation
from utils.mnist_loader import load_data

# ─── Config ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic" / "data"
LATENT_DIM = 512
PHA1_LATENT_DIM = 256
EPOCHS = 110
BATCH_SIZE = 1024
NUM_CLASSES = 4

parser = argparse.ArgumentParser()
parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--max-samples-per-class", type=int, default=None, help="Maximum samples per class to load from disk")
parser.add_argument("--samples-per-epoch-per-class", type=int, default=None, help="Random subset size per class each epoch")
parser.add_argument("--pbm-dir", type=str, default=None, help="Load P4 PBM images from this directory instead of .npy")
args = parser.parse_args()

NUM_CLASSES = args.num_classes
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
MAX_SAMPLES_PER_CLASS = args.max_samples_per_class
SAMPLES_PER_EPOCH_PER_CLASS = args.samples_per_epoch_per_class
PBM_DIR = Path(args.pbm_dir) if args.pbm_dir else None

#PROGRESSIVE_EPOCHS = 36 # Phase 1: 14x14
PROGRESSIVE_EPOCHS = 24 # Phase 1: 14x14
LR = 0.003
LR_WARMUP_START = 0.00005
LR_WARMUP_EPOCHS = 5
LR_DECAY_EPOCHS = EPOCHS - LR_WARMUP_EPOCHS
LR_END = 0.0001
INFO_NCE_WEIGHT = 0.001
TEMPERATURE = 1.0
# Foreground weight for pixel BCE. Sparse classes (e.g. class 4, ~12.6%
# foreground) are dominated by background under plain BCE, which drags
# the generator output below the 0.5 threshold. (1-fg)/fg ~= 7 balances
# the positive/negative contributions at the measured coverage.
PIXEL_POS_WEIGHT = 2.0
DROPOUT = 0.3
SPATIAL_DROPOUT = 0.0
L2_REG = 0.00001
# VAE-GAN hybrid (Phase 2 only): adversarial term + discriminator LR
ADV_LAMBDA = 0.4
DISC_LR = 0.00005
D_BATCH_INTERVAL = 2  # update discriminator every N batches (G updates every batch)
D_REAL_LABEL = 1.0    # one-sided label smoothing for the real class
# Phase 2 KL schedule (configurable start)
PHA2_KL_WARMUP_START = 0.0
PHA2_KL_WEIGHT_START = 0.04
PHA2_KL_WEIGHT_TARGET = 0.0002
PHA2_KL_WARMUP_EPOCHS = 20
PHA2_KL_DECAY_EPOCHS = 50 - PROGRESSIVE_EPOCHS
# Phase 2 pixel-loss schedule. Pixel BCE is what pulls the generator toward the
# per-pixel mean (blur); annealing it down lets the perceptual + adversarial
# terms sharpen instead. Linear, no warmup: full weight at the start of phase 2,
# TARGET on the last epoch actually run -- hence the -1, since phase 2 runs
# epochs PROGRESSIVE_EPOCHS .. EPOCHS-1.
PHA2_PIXEL_WEIGHT_START = 1.0 * 20
PHA2_PIXEL_WEIGHT_TARGET = 0.1 * 20
PHA2_PIXEL_DECAY_EPOCHS = EPOCHS - PROGRESSIVE_EPOCHS - 1
# Phase 2 perceptual-loss schedule. The perceptual term is measured through a
# FROZEN MNIST classifier, so early in phase 2 the generator's output is far
# enough from real digits that its features carry no useful signal -- a strong
# perceptual gradient then pulls toward the classifier's notion of "digit"
# before the basic structure exists. So: 0 at the start of phase 2, warm up once
# the pixel term has established coarse structure, then decay.
# Note the perceptual loss is a feature-MSE (~0.015), roughly 2 orders of
# magnitude below the pixel/KL terms, so these weights are correspondingly large.
PHA2_PERC_WARMUP_START = 0.0
PHA2_PERC_WEIGHT_START = 300.0   # peak, reached at the end of warmup
PHA2_PERC_WEIGHT_TARGET = 2.0
PHA2_PERC_WARMUP_EPOCHS = 35
PHA2_PERC_DECAY_EPOCHS = EPOCHS - PROGRESSIVE_EPOCHS - PHA2_PERC_WARMUP_EPOCHS - 1 - 10
PHA2_TRANSFER_FREEZE_EPOCHS = 1
# Phase 1 (14x14) specific config
PHA1_LR = 0.005
PHA1_LR_WARMUP_START = 0.00005
PHA1_LR_WARMUP_EPOCHS = 8
PHA1_LR_END = 0.0001
PHA1_LR_DECAY_EPOCHS = PROGRESSIVE_EPOCHS - PHA1_LR_WARMUP_EPOCHS
PHA1_PIXEL_LOSS_WEIGHT = 1.0 * 20
PHA1_KL_WARMUP_EPOCHS = PROGRESSIVE_EPOCHS - 1
# start
PHA1_KL_WARMUP_START = 0.003
# end
PHA1_KL_WEIGHT_START = 0.006
PHA1_INFO_NCE_WEIGHT = 0.001
LOG_NAME = 'logs/gen1b'
LOG_NAME_PHA1 = 'logs/gen1b/phase1'

def weighted_pixel_loss(target, reconstruction):
    """Foreground-weighted pixel BCE, as a PER-PIXEL MEAN.

    keras binary_crossentropy returns shape [B,H,W] (the channel axis is
    squeezed), while `target` is [B,H,W,1] -- so the weights are squeezed
    to match before multiplying.

    This previously summed over (H, W) and averaged over the batch, so its
    value scaled with the pixel count: ~196 at 14x14 but ~784 at 28x28, and
    ~5 orders of magnitude larger than the perceptual loss (which is a mean).
    It is now a mean over pixels, so it is O(1) and on the same footing as the
    perceptual / adversarial / KL terms.

    NOTE: this shrinks the pixel term by H*W, so the pixel:KL balance shifts by
    that same factor and the loss weights need retuning to compensate.
    """
    bce = keras.losses.binary_crossentropy(target, reconstruction)
    t = tf.squeeze(target, axis=-1)
    pixel_weights = t * PIXEL_POS_WEIGHT + (1.0 - t)
    return tf.reduce_mean(tf.reduce_mean(bce * pixel_weights, axis=(1, 2)))

# ─── Data ──────────────────────────────────────────────────────────
def load_data_cvae():
    (x_train, y_train), (x_test, y_test) = load_data(DATA_DIR, NUM_CLASSES, flip_pixels=7*28, max_samples_per_class=MAX_SAMPLES_PER_CLASS, pbm_dir=PBM_DIR, synthetic_last_class=False)
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data_cvae()


# ─── Resize helper for progressive growing ─────────────────────────
def resize_images(images, target_size):
    """Resize images to target_size using bilinear interpolation."""
    return tf.image.resize(images, target_size, method='bilinear').numpy()


# ─── Sampling Layer ────────────────────────────────────────────────
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ─── Encoder ──────────────────────────────────────────────────────
def build_encoder():
    img = keras.Input(shape=(28, 28, 1), name="image_input")
    lbl = keras.Input(shape=(NUM_CLASSES,), name="label_input")

    # Label-conditioned spatial gate (CBAM-style). Channel-pool (avg+max) +
    # broadcast label -> kxk conv -> sigmoid gate. Returns the GATE (not
    # x * gate) so it can be fused with FiLM by CollabModulation at c1 -- the
    # same contract as build_encoder_14 and the generator's
    # spatial_attention_map.
    def spatial_attention_map(x, lbl, k, name=None):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx = tf.reduce_max(x, axis=-1, keepdims=True)
        pooled = layers.Concatenate()([avg, mx])
        h = int(x.shape[1])
        lbl_map = layers.Reshape((1, 1, NUM_CLASSES))(lbl)
        lbl_map = layers.UpSampling2D(size=h, interpolation="nearest")(lbl_map)
        att_in = layers.Concatenate()([pooled, lbl_map])
        att = layers.Conv2D(1, k, padding="same", activation="sigmoid", name=name)(att_in)
        return att

    b3 = layers.Conv2D(32, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc28_conv3")(img)
    b5 = layers.Conv2D(32, 5, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc28_conv5")(img)
    b7 = layers.Conv2D(32, 7, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc28_conv7")(img)
    x28_main = layers.Concatenate(name="enc28_cat")([b3, b5, b7])   # 3 x ch @ 28x28
    x28_main = layers.LayerNormalization(name="enc28_ln")(x28_main)
    # Residual shortcut for the front-end. Requires a LEARNED 1x1 projection:
    # the raw image has 1 channel and x28_main has 48. The old removed version
    # added the same scalar to every channel, so LayerNorm's per-pixel mean
    # subtraction cancelled it exactly (measured 9.5e-07); a 1x1 projection
    # gives each channel its own weight, so the mean shift is not uniform and
    # the shortcut survives (measured 0.35). Phase-2 only.
    sc28 = layers.Conv2D(96, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc28_proj")(img)
    x28_main = layers.Add()([x28_main, sc28])
    x28_main = layers.Activation("relu")(x28_main)

    # Conv block 1: 28x28 → 14x14
    c1 = layers.Conv2D(96, 3, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv1")(x28_main)
    c1 = tf.keras.layers.LayerNormalization(name='enc_14_ln')(c1)
    # Residual shortcut, ResNet-D style: enc_conv1 and the avg-pooled front-end
    # both map 28x28x48 -> 14x14x48, so no spatial reshape is needed and the
    # 1x1 conv lets the network learn the shortcut instead of forcing a plain
    # blur. Phase-2 only -- phase 1 has no 28x28 scale, so this name has no
    # counterpart and is deliberately not transferred.
    sc = layers.Conv2D(96, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc28_skip_proj")(layers.AveragePooling2D(2)(x28_main))
    c1 = layers.Add()([c1, sc])
    c1 = layers.Activation("relu")(c1)

    att = spatial_attention_map(c1, lbl, k=5, name="enc_spatial_attn")
    gamma14 = layers.Reshape((1, 1, 96))(2.0 * layers.Dense(96, activation="sigmoid", name="enc_film14_g")(lbl))
    beta14 = layers.Reshape((1, 1, 96))(layers.Dense(96, name="enc_film14_b")(lbl))
    c1 = CollabModulation(name="enc_collab14")([c1, gamma14, att, beta14])

    # Conv block 2: 14x14 → 7x7
    c2 = layers.Conv2D(192, 3, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv2")(c1)
    c2 = layers.Activation("relu")(c2)
    # Residual connection: project c1 (14x14x64) to match c2 (7x7x128)
    skip = layers.Conv2D(192, 1, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv2_skip")(c1)
    c2 = layers.Add()([c2, skip])
    c2 = layers.Activation("relu")(c2)

    # Cross-scale shortcut from c1 into block 3
    pooled_c1 = layers.AveragePooling2D(2)(c1)
    c2 = layers.Concatenate()([c2, pooled_c1])

    # Conv block 3: 7x7 -> 7x7 (stride 1). Named for the phase-1 -> phase-2
    # transfer contract: enc_conv3 and enc_conv3_skip cannot transfer (their
    # input width differs by phase), but enc_se3_a/b and enc_se3_skip have identical
    # shapes in both and do transfer.
    c3 = layers.Conv2D(192, 3, strides=1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv3")(c2)
    c3 = layers.Activation("relu")(c3)
    # Residual connection: 1x1 projection of c2 to c3's channel count
    skip2 = layers.Conv2D(192, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv3_skip")(c2)
    c3 = layers.Add()([c3, skip2])
    c3 = layers.Activation("relu")(c3)
    # SE attention 3
    se3 = layers.GlobalAveragePooling2D()(c3)
    se3 = layers.Dense(192 // 4, activation="relu", name="enc_se3_a")(se3)
    se3 = layers.Dense(192, activation="sigmoid", name="enc_se3_b")(se3)
    se3 = layers.Reshape((1, 1, 192))(se3)
    c3_pre = c3
    c3 = layers.Multiply()([c3, se3])
    c3 = layers.Activation("relu")(c3)
    skip3 = layers.Conv2D(192, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_se3_skip")(c3_pre)
    c3 = layers.Add()([c3, skip3])

    # each contributes ~equally to the latent instead of fine being ~2%.
    main = layers.Flatten()(c3)
    main = layers.Concatenate()([main, lbl])
    main = layers.Dense(512, activation="relu", name="enc28_head_dense", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(main)
    main = layers.LayerNormalization(name="enc28_head_ln")(main)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(main)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(main)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model([img, lbl], [z_mean, z_log_var, z], name="encoder")


# ─── Encoder for 14×14 Phase 1 ─────────────────────────────────────
def build_encoder_14():
    img = keras.Input(shape=(14, 14, 1), name="image_input")
    lbl = keras.Input(shape=(NUM_CLASSES,), name="label_input")

    # Label-conditioned spatial gate. Returns the GATE (not x * gate) so it can
    # be fused with FiLM by CollabModulation at c1 -- the same contract as the
    # generator's spatial_attention_map. The Conv2D keeps its name, so
    # enc_spatial_attn still matches phase 2 by name and shape.
    def spatial_attention_map(x, lbl, k, name=None):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx = tf.reduce_max(x, axis=-1, keepdims=True)
        pooled = layers.Concatenate()([avg, mx])
        h = int(x.shape[1])
        lbl_map = layers.Reshape((1, 1, NUM_CLASSES))(lbl)
        lbl_map = layers.UpSampling2D(size=h, interpolation="nearest")(lbl_map)
        att_in = layers.Concatenate()([pooled, lbl_map])
        att = layers.Conv2D(1, k, padding="same", activation="sigmoid", name=name)(att_in)
        return att

    # 14x14 front-end, phase 1. Mirrors the enc28_* front-end in build_encoder,
    # but 3/5 rather than 3/5/7: bilinear downsampling to 14x14 halves stroke
    # thickness, so the scale-appropriate kernel sizes are smaller. This block
    # does NOT transfer -- phase 2's 28x28 front-end plays this role there.
    # Fixes vs the old x14_main: (a) the Add was erased by the LayerNorm (the
    # image is added identically to every channel, so LN's per-pixel mean
    # subtraction cancels it exactly) -- it is gone; (b) relu -> LN -> relu
    # clipped the pre-activations twice around the norm -- now LN sees raw
    # pre-activations with a single relu after, exactly as in phase 2.
    b3 = layers.Conv2D(48, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc14_conv3")(img)
    b5 = layers.Conv2D(48, 5, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc14_conv5")(img)
    x14_main = layers.Concatenate(name="enc14_cat")([b3, b5])   # 2 x ch @ 14x14
    x14_main = layers.LayerNormalization(name="enc14_ln")(x14_main)
    x14_main = layers.Activation("relu")(x14_main)

    sc14 = layers.Conv2D(96, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="ph1_enc14_proj")(img)
    x14 = layers.Add()([x14_main, sc14])
    x14 = layers.Activation("relu")(x14)

    # Label injection at c1: per-channel FiLM fused with the spatial gate via
    # learned colla-factors (CAT, collab.py):
    #     x' = x * (c_w * gamma + s_w * att) + beta
    # c_w=1, s_w=0 at init, so this starts as pure FiLM -- a direct linear
    # 5->64 map from the label -- instead of only shifting a sigmoid's operating
    # point, which is all the old `x * att` could do: lbl_map is spatially
    # constant, so the label carried no positional information and the gate
    # varied only by a per-class gain.
    # Phase-1 only, deliberately: the generator already uses this mechanism at
    # 14x14/28x28, so encoder and generator conditioning are now symmetric.
    # enc_film14_*/enc_collab14 are phase-1-only names and do not transfer.
    att = spatial_attention_map(x14, lbl, k=5, name="enc_spatial_attn")
    gamma14 = layers.Reshape((1, 1, 96))(2.0 * layers.Dense(96, activation="sigmoid", name="enc_film14_g")(lbl))
    beta14 = layers.Reshape((1, 1, 96))(layers.Dense(96, name="enc_film14_b")(lbl))
    c1 = CollabModulation(name="enc_collab14")([x14, gamma14, att, beta14])

    # Block 2: 14x14 -> 7x7 (mirrored from 28x28 encoder block 2)
    c2 = layers.Conv2D(192, 3, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv2")(c1)
    c2 = layers.Activation("relu")(c2)
    skip = layers.Conv2D(192, 1, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv2_skip")(c1)
    c2 = layers.Add()([c2, skip])
    c2 = layers.Activation("relu")(c2)

    # Cross-scale shortcut from c1 into block 3
    pooled_c1 = layers.AveragePooling2D(2)(c1)
    c2 = layers.Concatenate()([c2, pooled_c1])

    # Conv block 3: 7x7 -> 7x7 (stride 1). Named for the phase-1 -> phase-2
    # transfer contract: enc_conv3 and enc_conv3_skip cannot transfer (their
    # input width differs by phase -- the c2 concat is 192ch in phase 1 and
    # 144ch in phase 2), but enc_se3_a/b and enc_se3_skip have identical
    # shapes in both and do transfer.
    c3 = layers.Conv2D(192, 3, strides=1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv3")(c2)
    c3 = layers.Activation("relu")(c3)
    # Residual connection: 1x1 projection of c2 to c3's channel count
    skip2 = layers.Conv2D(192, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv3_skip")(c2)
    c3 = layers.Add()([c3, skip2])
    c3 = layers.Activation("relu")(c3)
    # SE attention 3
    se3 = layers.GlobalAveragePooling2D()(c3)
    se3 = layers.Dense(192 // 4, activation="relu", name="enc_se3_a")(se3)
    se3 = layers.Dense(192, activation="sigmoid", name="enc_se3_b")(se3)
    se3 = layers.Reshape((1, 1, 192))(se3)
    c3_pre = c3
    c3 = layers.Multiply()([c3, se3])
    c3 = layers.Activation("relu")(c3)
    skip3 = layers.Conv2D(192, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_se3_skip")(c3_pre)
    c3 = layers.Add()([c3, skip3])

    main = layers.Flatten()(c3)
    main = layers.Concatenate()([main, lbl])
    main = layers.Dense(256, activation="relu", name="enc14_head_dense", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(main)
    main = layers.LayerNormalization(name="enc14_head_ln")(main)

    z_mean = layers.Dense(PHA1_LATENT_DIM, name="z_mean", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(main)
    z_log_var = layers.Dense(PHA1_LATENT_DIM, name="z_log_var", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(main)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model([img, lbl], [z_mean, z_log_var, z], name="encoder_14")


# ─── Upsample via Sub-pixel Convolution ────────────────────────────
def upsample(x, target_size, name=None):
    """Upsample using sub-pixel convolution (depth_to_space).

    `name` is required for name-based weight transfer: without it the conv is
    auto-named, and the auto-name differs between the phase-1 and phase-2
    builders (it depends on how many layers were created earlier in the
    process), so there would be nothing to match on.
    """
    channels = x.shape[-1]
    x = layers.Conv2D(channels * 4, 3, padding='same', activation='relu', name=name)(x)
    x = tf.nn.depth_to_space(x, 2)
    return x


# ─── Generator for 14×14 Phase 1 ────────────────────────────────────
def build_generator_14():
    z = keras.Input(shape=(PHA1_LATENT_DIM,), name="latent_input")
    lbl = keras.Input(shape=(NUM_CLASSES,), name="label_input")

    def make_film(channels, tag):
        def film_fn(label):
            gamma = 2.0 * layers.Dense(channels, activation="sigmoid", name=f"{tag}_g")(label)  # [0,2], init ~1 (identity)
            beta = layers.Dense(channels, name=f"{tag}_b")(label)
            return gamma, beta
        return film_fn

    # Spatial attention map (mirrors build_generator for positional transfer).
    # Returns the gate so it can be combined with FiLM via colla-factors (CAT).
    def spatial_attention_map(x, lbl, k, name=None):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx = tf.reduce_max(x, axis=-1, keepdims=True)
        pooled = layers.Concatenate()([avg, mx])
        h = int(x.shape[1])
        lbl_map = layers.Reshape((1, 1, NUM_CLASSES))(lbl)
        lbl_map = layers.UpSampling2D(size=h, interpolation="nearest")(lbl_map)
        att_in = layers.Concatenate()([pooled, lbl_map])
        att = layers.Conv2D(1, k, padding="same", activation="sigmoid", name=name)(att_in)
        return att

    # Names below are the transfer contract: every layer whose role exists at
    # the same resolution in build_generator carries the identical name, so
    # transfer_weights_by_name copies exactly those. Layers that are phase-1
    # only (the 14x14 output conv) carry a phase-specific name and are
    # deliberately left untransferred.
    #
    # Initial projection to 7x7 (kernel shapes differ between the phases --
    # latent dims differ -- so only its bias transfers).
    x = layers.Dense(7 * 7 * 256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_proj7")(z)
    x = layers.Reshape((7, 7, 256))(x)
    # Skip at the BEGINNING of the 7x7 scale
    skip_7 = layers.Conv2D(128, 1, padding="same", name="gen_skip7")(x)
    # 7x7 block (single FiLM after conv; seed FiLM dropped)
    x = layers.Conv2D(256, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_conv7")(x)
    gamma7c, beta7c = make_film(256, "gen_film7")(lbl)
    gamma7c = layers.Reshape((1, 1, 256))(gamma7c)
    beta7c = layers.Reshape((1, 1, 256))(beta7c)
    x = layers.Multiply()([x, gamma7c])
    x = layers.Add()([x, beta7c])
    # SE at the 7x7 bottleneck (one consolidated)
    se7 = layers.GlobalAveragePooling2D()(x)
    se7 = layers.Dense(256 // 4, activation="relu", name="gen_se7_a")(se7)
    se7 = layers.Dense(256, activation="sigmoid", name="gen_se7_b")(se7)
    se7 = layers.Reshape((1, 1, 256))(se7)
    x = layers.Multiply()([x, se7])

    # 7x7 → 14x14. Narrow BEFORE upsampling: the sub-pixel conv costs
    # 36*H*W*C^2, so doing the 256->128 reduction at 7x7 rather than at 14x14
    # drops this step from 173.4M to 43.4M MACs. gen_conv14 keeps its shape
    # (3,3,256,128), so the phase-1 -> phase-2 transfer is unaffected.
    x = layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_conv14")(x)
    x = upsample(x, [14, 14], name="gen_up7")
    gamma14, beta14 = make_film(128, "gen_film14")(lbl)
    gamma14 = layers.Reshape((1, 1, 128))(gamma14)
    beta14 = layers.Reshape((1, 1, 128))(beta14)
    # CAT collaboration at 14x14 (k=5)
    att14 = spatial_attention_map(x, lbl, k=5, name="spatial_attn_14")
    x = CollabModulation(name="collab_14")([x, gamma14, att14, beta14])
    # Fusion instead of Add: Concatenate keeps the two streams separate and the
    # 1x1 conv learns an arbitrary per-channel mix of them (Add is the special
    # case where that mixing matrix is hard-wired to [I | I]). LayerNorm
    # afterwards follows the conv -> norm -> relu convention used elsewhere in
    # this file. gen_skip7_fuse/gen_skip7_ln carry the same names in
    # build_generator, so they transfer phase 1 -> phase 2.
    skip_7_up = upsample(skip_7, [14, 14], name="gen_up7_skip")
    x = layers.Concatenate(name="gen_skip7_cat")([x, skip_7_up])       # 256 @ 14x14
    x = layers.Conv2D(128, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_skip7_fuse")(x)
    x = layers.LayerNormalization(name="gen_skip7_ln")(x)
    x = layers.Activation("relu")(x)

    # Output conv at 14x14. Phase 2's equivalent is gen_out28 at 28x28 with a
    # different input width, so there is no name match and no transfer.
    x = layers.Conv2D(1, 3, padding="same", activation="sigmoid", dtype="float32", name="gen_out14")(x)
    return keras.Model([z, lbl], x, name="generator_14")


# ─── Generator with FiLM Conditioning ──────────────────────────────
def build_generator():
    z = keras.Input(shape=(LATENT_DIM,), name="latent_input")
    lbl = keras.Input(shape=(NUM_CLASSES,), name="label_input")

    # FiLM helper: produce scale (gamma) and shift (beta) from label
    def make_film(channels, tag):
        def film_fn(label):
            gamma = 2.0 * layers.Dense(channels, activation="sigmoid", name=f"{tag}_g")(label)  # [0,2], init ~1 (identity)
            beta = layers.Dense(channels, name=f"{tag}_b")(label)
            return gamma, beta
        return film_fn

    # Spatial attention map (CBAM-style): channel-pool (avg+max) + broadcast
    # label -> kxk conv -> sigmoid gate. Returns the gate (not x*att) so it
    # can be combined with FiLM via learned colla-factors (CAT).
    def spatial_attention_map(x, lbl, k, name=None):
        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        mx = tf.reduce_max(x, axis=-1, keepdims=True)
        pooled = layers.Concatenate()([avg, mx])                  # (B,H,W,2)
        h = int(x.shape[1])
        lbl_map = layers.Reshape((1, 1, NUM_CLASSES))(lbl)
        lbl_map = layers.UpSampling2D(size=h, interpolation="nearest")(lbl_map)
        att_in = layers.Concatenate()([pooled, lbl_map])          # (B,H,W,2+NC)
        att = layers.Conv2D(1, k, padding="same", activation="sigmoid", name=name)(att_in)
        return att

    # Names are the transfer contract with build_generator_14: every layer
    # whose role exists at the same resolution in phase 1 carries the identical
    # name. Layers below marked "phase-2 only" have no phase-1 counterpart and
    # are deliberately left untransferred.
    #
    # Initial projection (kernel shapes differ -- latent dims differ -- so only
    # its bias transfers)
    x = layers.Dense(7 * 7 * 256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_proj7")(z)
    x = layers.Reshape((7, 7, 256))(x)
    # Skip saved at the BEGINNING of the 7x7 scale (freshest features, before
    # the conv/FiLM/attention stack processes them)
    skip_7 = layers.Conv2D(128, 1, padding="same", name="gen_skip7")(x)
    # 7x7 block (single FiLM after conv; seed FiLM dropped to reduce over-conditioning)
    x = layers.Conv2D(256, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_conv7")(x)
    # FiLM at 7x7 (after conv)
    gamma7c, beta7c = make_film(256, "gen_film7")(lbl)
    gamma7c = layers.Reshape((1, 1, 256))(gamma7c)
    beta7c = layers.Reshape((1, 1, 256))(beta7c)
    x = layers.Multiply()([x, gamma7c])
    x = layers.Add()([x, beta7c])
    # SE at the 7x7 bottleneck (one consolidated; high channels + semantic)
    se7 = layers.GlobalAveragePooling2D()(x)
    se7 = layers.Dense(256 // 4, activation="relu", name="gen_se7_a")(se7)
    se7 = layers.Dense(256, activation="sigmoid", name="gen_se7_b")(se7)
    se7 = layers.Reshape((1, 1, 256))(se7)
    x = layers.Multiply()([x, se7])
    # Narrow BEFORE upsampling (see build_generator_14): 256->128 at 7x7
    # instead of at 14x14 drops this step from 173.4M to 43.4M MACs.
    x = layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_conv14")(x)
    x = upsample(x, [14, 14], name="gen_up7")
    # Skip saved at the BEGINNING of the 14x14 scale. Phase-2 only: it feeds
    # the 28x28 stage, which phase 1 does not have. It now taps the narrowed
    # 128ch tensor (was 256ch before the reduction moved to 7x7), so its shape
    # becomes (1,1,128,64) instead of (1,1,256,64).
    skip_14 = layers.Conv2D(64, 1, padding="same", name="gen_skip14")(x)
    # FiLM at 14x14
    gamma14, beta14 = make_film(128, "gen_film14")(lbl)
    gamma14 = layers.Reshape((1, 1, 128))(gamma14)
    beta14 = layers.Reshape((1, 1, 128))(beta14)
    # CAT collaboration: learned colla-factors combine FiLM (channel) + spatial attn
    att14 = spatial_attention_map(x, lbl, k=5, name="spatial_attn_14")
    x = CollabModulation(name="collab_14")([x, gamma14, att14, beta14])
    # Residual: project skip_7 to 14x14, then fuse by concatenation rather than
    # Add (see build_generator_14). Same names as phase 1, so gen_skip7_fuse and
    # gen_skip7_ln transfer in from phase 1.
    skip_7_up = upsample(skip_7, [14, 14], name="gen_up7_skip")
    x = layers.Concatenate(name="gen_skip7_cat")([x, skip_7_up])       # 256 @ 14x14
    x = layers.Conv2D(128, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_skip7_fuse")(x)
    x = layers.LayerNormalization(name="gen_skip7_ln")(x)
    x = layers.Activation("relu")(x)

    # ─── 28x28 stage: phase-2 only, no phase-1 counterpart ──────────────
    x = layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_conv14b")(x)
    # FiLM for the 28x28 stage (applied at 14x14, before the upsample)
    gamma28, beta28 = make_film(64, "gen_film14b")(lbl)
    gamma28 = layers.Reshape((1, 1, 64))(gamma28)
    beta28 = layers.Reshape((1, 1, 64))(beta28)
    # CAT collaboration at 28x28 (k=7). No SE here (redundant with output conv).
    att28 = spatial_attention_map(x, lbl, k=7, name="spatial_attn_28")
    x = CollabModulation(name="collab_28")([x, gamma28, att28, beta28])
    x = upsample(x, [28, 28], name="gen_up14")
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="gen_conv28")(x)
    # Same concat + 1x1 fuse + LayerNorm treatment as the 14x14 skip. This one
    # is phase-2 only (no 28x28 stage in phase 1), so gen_skip14_* have no
    # counterpart and are deliberately not transferred.
    skip_14_up = upsample(skip_14, [28, 28], name="gen_up14_skip")
    x = layers.Concatenate(name="gen_skip14_cat")([x, skip_14_up])     # 128 @ 28x28
    x = layers.Conv2D(64, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_skip14_fuse")(x)
    x = layers.LayerNormalization(name="gen_skip14_ln")(x)
    x = layers.Activation("relu")(x)

    # Output conv at 28x28 (phase 1's equivalent is gen_out14 at 14x14)
    x = layers.Conv2D(1, 3, padding="same", activation="sigmoid", dtype="float32", name="gen_out28")(x)
    return keras.Model([z, lbl], x, name="generator")


# ─── Perceptual Loss Feature Extractor (Multi-layer) ───────────────
def build_feature_extractor():
    """Use the trained MNIST classifier as fixed semantic multi-layer feature extractor."""
    mnist_path = Path(__file__).resolve().parent / "mnist_model"
    if not mnist_path.exists():
        raise FileNotFoundError(f"Required perceptual backbone not found: {mnist_path}")
    custom_objects={'loss_fn': lambda y_true, y_pred: keras.losses.categorical_crossentropy(y_true, y_pred)}
    cls_model = keras.models.load_model(str(mnist_path), custom_objects=custom_objects)
    # Fixed semantic points: 28x28, 14x14, 7x7, and dense-style pool features
    layer_names = [
        'percept_c0',
        'percept_c1',
        'percept_c2',
        'percept_c3',
        'pool7',
        'pool14',
        'pool28',
    ]
    feature_layers = []
    for name in layer_names:
        try:
            layer = cls_model.get_layer(name)
            feature_layers.append(layer.output)
        except ValueError:
            pass
    if not feature_layers:
        # Fallback: use multi-layer conv extraction if custom named layers are unavailable
        for layer in cls_model.layers:
            if isinstance(layer, (layers.Conv2D, layers.SeparableConv2D)):
                feature_layers.append(layer.output)
    pooled = []
    for f in feature_layers:
        if len(f.shape) == 4:
            pooled.append(layers.GlobalAveragePooling2D()(f))
        else:
            pooled.append(f)
    feat_model = keras.Model(cls_model.input, pooled, name="feature_extractor")
    return feat_model


# ─── VAE Model ────────────────────────────────────────────────────
class VAE(keras.Model):
    def __init__(self, encoder, generator, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.generator = generator
        self.feature_extractor = build_feature_extractor()
        # Freeze feature extractor weights
        self.feature_extractor.trainable = False
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.kl_weight_tracker = keras.metrics.Mean(name="kl_weight")
        self.pixel_weight_tracker = keras.metrics.Mean(name="pixel_weight")
        self.perceptual_loss_tracker = keras.metrics.Mean(name="perceptual_loss")
        self.perceptual_weight_tracker = keras.metrics.Mean(name="perceptual_weight")
        self.epoch_tracker = tf.Variable(0, trainable=False, dtype=tf.int32)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_tracker.assign(epoch)

    def get_lr(self):
        epoch = tf.cast(self.epoch_tracker, tf.float32)
        warmup_epochs = tf.cast(LR_WARMUP_EPOCHS, tf.float32)
        decay_epochs = tf.cast(LR_DECAY_EPOCHS, tf.float32)
        # Warmup phase
        warmup_progress = epoch / warmup_epochs
        warmup_lr = LR_WARMUP_START + (LR - LR_WARMUP_START) * tf.minimum(1.0, warmup_progress)
        # Decay phase (linear)
        decay_epoch = epoch - warmup_epochs
        decay_progress = tf.minimum(1.0, decay_epoch / decay_epochs)
        decay_lr = LR + (LR_END - LR) * decay_progress
        return tf.where(epoch < warmup_epochs, warmup_lr, decay_lr)

    def get_kl_weight(self):
        epoch = tf.cast(self.epoch_tracker, tf.float32)
        kl_epoch = epoch - tf.cast(PROGRESSIVE_EPOCHS, tf.float32)
        warmup_epochs = tf.cast(PHA2_KL_WARMUP_EPOCHS, tf.float32)
        decay_epochs = tf.cast(PHA2_KL_DECAY_EPOCHS, tf.float32)
        start = tf.constant(PHA2_KL_WARMUP_START, dtype=tf.float32)
        mid = tf.constant(PHA2_KL_WEIGHT_START, dtype=tf.float32)
        end = tf.constant(PHA2_KL_WEIGHT_TARGET, dtype=tf.float32)
        zero = tf.constant(0.0, dtype=tf.float32)
        warmup_progress = tf.clip_by_value(kl_epoch / warmup_epochs, 0.0, 1.0)
        warmup_weight = start + (mid - start) * warmup_progress
        decay_epoch = kl_epoch - warmup_epochs
        decay_progress = tf.clip_by_value(decay_epoch / decay_epochs, 0.0, 1.0)
        decay_weight = mid + (end - mid) * decay_progress
        # Phase 1 (kl_epoch < 0): constant start
        # Phase 2 (0 <= kl_epoch < warmup): warmup
        # Phase 3 (kl_epoch >= warmup): decay
        is_phase1 = tf.cast(kl_epoch < zero, tf.float32)
        is_phase2 = tf.cast(tf.logical_and(kl_epoch >= zero, kl_epoch < warmup_epochs), tf.float32)
        is_phase3 = tf.cast(kl_epoch >= warmup_epochs, tf.float32)
        return start * is_phase1 + warmup_weight * is_phase2 + decay_weight * is_phase3

    def get_pixel_weight(self):
        """Linear anneal of the pixel-loss weight across phase 2.

        Pixel BCE is what drives the generator toward the per-pixel mean, so
        holding it constant keeps the output blurry no matter how strong the
        perceptual/adversarial terms get. Phase 1 deliberately has no schedule:
        its loss is pixel + a negligible KL + info_nce=0, so pixel is the only
        active term and decaying its weight would just rescale the whole loss --
        indistinguishable from decaying the LR.

        Progress is clipped at BOTH ends, so pre-phase-2 epochs clamp to START
        instead of overshooting above it. (get_lr/get_kl_weight use an
        asymmetric tf.minimum(1.0, ...) and rely on tf.where to mask the
        out-of-range branch; no need to repeat that trap here.)
        """
        epoch = tf.cast(self.epoch_tracker, tf.float32)
        p2_epoch = epoch - tf.cast(PROGRESSIVE_EPOCHS, tf.float32)
        start = tf.constant(PHA2_PIXEL_WEIGHT_START, dtype=tf.float32)
        end = tf.constant(PHA2_PIXEL_WEIGHT_TARGET, dtype=tf.float32)
        progress = tf.clip_by_value(
            p2_epoch / tf.cast(PHA2_PIXEL_DECAY_EPOCHS, tf.float32), 0.0, 1.0)
        return start + (end - start) * progress

    def get_perceptual_weight(self):
        """Warm up then decay the perceptual-loss weight across phase 2.

        The perceptual term is measured through a frozen MNIST classifier, so at
        the start of phase 2 the generator's output is far enough from real
        digits that the classifier's features are uninformative -- a strong
        perceptual gradient there pulls toward "what a digit looks like" before
        the basic structure exists. Ramp in after the pixel term has done its
        work, then decay so it does not fight the fine detail at the end.

        Same shape as get_kl_weight: constant `start` before phase 2, linear
        warmup start->mid over PHA2_PERC_WARMUP_EPOCHS, then linear decay
        mid->end. Continuous at the handover (both branches equal mid), and both
        progress terms are clipped at both ends.
        """
        epoch = tf.cast(self.epoch_tracker, tf.float32)
        p2_epoch = epoch - tf.cast(PROGRESSIVE_EPOCHS, tf.float32)
        warmup_epochs = tf.cast(PHA2_PERC_WARMUP_EPOCHS, tf.float32)
        decay_epochs = tf.cast(PHA2_PERC_DECAY_EPOCHS, tf.float32)
        start = tf.constant(PHA2_PERC_WARMUP_START, dtype=tf.float32)
        mid = tf.constant(PHA2_PERC_WEIGHT_START, dtype=tf.float32)
        end = tf.constant(PHA2_PERC_WEIGHT_TARGET, dtype=tf.float32)
        warmup_progress = tf.clip_by_value(p2_epoch / warmup_epochs, 0.0, 1.0)
        warmup_weight = start + (mid - start) * warmup_progress
        decay_progress = tf.clip_by_value(
            (p2_epoch - warmup_epochs) / decay_epochs, 0.0, 1.0)
        decay_weight = mid + (end - mid) * decay_progress
        return tf.where(p2_epoch < warmup_epochs, warmup_weight, decay_weight)

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker,
                self.kl_weight_tracker, self.pixel_weight_tracker,
                self.perceptual_loss_tracker, self.perceptual_weight_tracker]

    def train_step(self, data):
        images, labels = data

        # Update LR with warmup
        self.optimizer.learning_rate = self.get_lr()

        # Get loss weights (all scheduled across phase 2)
        kl_weight = self.get_kl_weight()
        pixel_weight = self.get_pixel_weight()
        perceptual_weight = self.get_perceptual_weight()

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([images, labels])
            reconstruction = self.generator([z, labels])

            # Pixel-level reconstruction loss (BCE), foreground-weighted so
            # sparse classes aren't dominated by background pixels.
            # Resize target to match reconstruction size (no-op if same size)
            target = tf.image.resize(images, tf.shape(reconstruction)[1:3])
            pixel_loss = weighted_pixel_loss(target, reconstruction)

            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))

            # InfoNCE: ensure encoder utilizes latent space
            # For each batch, compute pairwise similarities
            batch_size = tf.shape(z)[0]
            z_norm = tf.math.l2_normalize(z, axis=1)
            # Similarity matrix: [batch, batch]
            sim_matrix = tf.matmul(z_norm, z_norm, transpose_b=True) / TEMPERATURE
            # Labels: diagonal (positive pairs)
            labels_ce = tf.range(batch_size)
            info_nce_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ce, logits=sim_matrix))

            # Perceptual loss (multi-layer feature-level)
            real_features = self.feature_extractor(images)
            fake_features = self.feature_extractor(reconstruction)
            # Mean over feature layers of the per-layer MSE. reduce_mean already
            # divides by the element count, so the previous `diff / n` divided by
            # it a second time -- and since n = batch x channels, that made the
            # term shrink as the batch grew. It also disagreed with test_step
            # (which never divided) by ~1600x. Both now compute the same thing.
            perceptual_loss = 0.0
            count = 0
            for rf, ff in zip(real_features, fake_features):
                perceptual_loss += tf.reduce_mean(tf.square(rf - ff))
                count += 1
            perceptual_loss /= tf.maximum(tf.cast(count, tf.float32), 1.0)

            # Combined loss
            total_loss = (pixel_weight * pixel_loss +
                          perceptual_weight * perceptual_loss +
                          kl_weight * kl_loss +
                          INFO_NCE_WEIGHT * info_nce_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(pixel_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_weight_tracker.update_state(kl_weight)
        self.pixel_weight_tracker.update_state(pixel_weight)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.perceptual_weight_tracker.update_state(perceptual_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        z_mean, z_log_var, z = self.encoder([images, labels])
        reconstruction = self.generator([z, labels])
        target = tf.image.resize(images, tf.shape(reconstruction)[1:3])
        pixel_loss = weighted_pixel_loss(target, reconstruction)
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        kl_weight = self.get_kl_weight()
        pixel_weight = self.get_pixel_weight()
        perceptual_weight = self.get_perceptual_weight()
        batch_size = tf.shape(z)[0]
        z_norm = tf.math.l2_normalize(z, axis=1)
        sim_matrix = tf.matmul(z_norm, z_norm, transpose_b=True) / TEMPERATURE
        labels_ce = tf.range(batch_size)
        info_nce_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ce, logits=sim_matrix))
        real_features = self.feature_extractor(images)
        fake_features = self.feature_extractor(reconstruction)
        perceptual_loss = tf.reduce_mean([
            tf.reduce_mean(tf.square(rf - ff))
            for rf, ff in zip(real_features, fake_features)
        ])
        total_loss = (pixel_weight * pixel_loss +
                      perceptual_weight * perceptual_loss +
                      kl_weight * kl_loss +
                      INFO_NCE_WEIGHT * info_nce_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(pixel_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_weight_tracker.update_state(kl_weight)
        self.pixel_weight_tracker.update_state(pixel_weight)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.perceptual_weight_tracker.update_state(perceptual_weight)
        return {m.name: m.result() for m in self.metrics}


# ─── Discriminator (conditional critic) ────────────────────────────
def build_discriminator():
    img = keras.Input(shape=(28, 28, 1), name="image_input")
    lbl = keras.Input(shape=(NUM_CLASSES,), name="label_input")

    # Broadcast label to a spatial channel map and concat (robust conditioning)
    lbl_map = layers.Reshape((1, 1, NUM_CLASSES))(lbl)
    lbl_map = layers.UpSampling2D(size=28, interpolation="nearest")(lbl_map)
    x = layers.Concatenate()([img, lbl_map])

    for filters in (64, 128, 256):
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    out = layers.Dense(1)(x)  # real/fake logit
    return keras.Model([img, lbl], out, name="discriminator")


# ─── VAE-GAN hybrid (Phase 2 only) ─────────────────────────────────
class VAEGAN(VAE):
    """VAE with a conditional discriminator and adversarial loss.

    The adversarial term forces the generator to commit to sharp output
    instead of the KL/BCE-averaged blur. Phase 1 stays pure VAE; the
    discriminator joins in Phase 2 (it does not transfer from Phase 1).
    """
    def __init__(self, encoder, generator, discriminator, **kwargs):
        super().__init__(encoder, generator, **kwargs)
        self.discriminator = discriminator
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")
        self.adv_loss_tracker = keras.metrics.Mean(name="adv_loss")
        self._d_step = tf.Variable(0, trainable=False, dtype=tf.int32)

    def compile(self, g_opt, d_opt, **kwargs):
        super().compile(optimizer=g_opt, **kwargs)
        self.d_opt = d_opt

    @property
    def metrics(self):
        return super().metrics + [self.disc_loss_tracker, self.adv_loss_tracker]

    def _bce(self, labels, logits):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits))

    def train_step(self, data):
        images, labels = data
        self.optimizer.learning_rate = self.get_lr()
        kl_weight = self.get_kl_weight()
        pixel_weight = self.get_pixel_weight()
        perceptual_weight = self.get_perceptual_weight()

        # Encode once for the discriminator's fake sample. Deliberately OUTSIDE
        # any tape: this pass needs no gradients, and recording it would build
        # a graph nobody differentiates. The encoder is called a second time
        # inside g_tape below -- see the note there for why that is required.
        _, _, z_d = self.encoder([images, labels])
        reconstruction_d = self.generator([z_d, labels])

        # --- Train discriminator every D_BATCH_INTERVAL batches ---
        # (updating G twice as often keeps D from collapsing to 0).
        # apply_gradients is called EVERY batch (Keras optimizers create
        # their variables lazily on first call, which must happen inside
        # the traced tf.function); the gate zeroes the update on off-batches.
        do_d_update = tf.cast(
            tf.equal(tf.math.floormod(self._d_step, D_BATCH_INTERVAL), 0),
            tf.float32)
        self._d_step.assign_add(1)
        with tf.GradientTape() as d_tape:
            d_real = self.discriminator([images, labels])
            d_fake = self.discriminator([tf.stop_gradient(reconstruction_d), labels])
            # one-sided label smoothing: real target 0.9 (not 1.0) so D
            # cannot reach 0 confidence and keeps providing gradient to G
            d_loss = (self._bce(D_REAL_LABEL * tf.ones_like(d_real), d_real)
                      + self._bce(tf.zeros_like(d_fake), d_fake))
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
        d_grads = [g * do_d_update if g is not None else g for g in d_grads]
        self.d_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))
        self.disc_loss_tracker.update_state(d_loss)

        # --- Train generator (encoder + decoder; adversarial flows through D) ---
        with tf.GradientTape() as g_tape:
            # The encoder MUST be called inside this tape. z_mean/z_log_var/z
            # feed kl_loss, info_nce_loss and the reconstruction; a forward pass
            # executed outside the tape is a constant to it, so every encoder
            # gradient would come back None and apply_gradients would skip it --
            # the encoder would silently never update in phase 2.
            z_mean, z_log_var, z = self.encoder([images, labels])
            reconstruction = self.generator([z, labels])
            d_fake = self.discriminator([reconstruction, labels])
            adv_loss = self._bce(tf.ones_like(d_fake), d_fake)

            target = tf.image.resize(images, tf.shape(reconstruction)[1:3])
            pixel_loss = weighted_pixel_loss(target, reconstruction)
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            batch_size = tf.shape(z)[0]
            z_norm = tf.math.l2_normalize(z, axis=1)
            sim_matrix = tf.matmul(z_norm, z_norm, transpose_b=True) / TEMPERATURE
            labels_ce = tf.range(batch_size)
            info_nce_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ce, logits=sim_matrix))
            real_features = self.feature_extractor(images)
            fake_features = self.feature_extractor(reconstruction)
            # Mean over feature layers of the per-layer MSE. reduce_mean already
            # divides by the element count, so the previous `diff / n` divided by
            # it a second time -- and since n = batch x channels, that made the
            # term shrink as the batch grew. It also disagreed with test_step
            # (which never divided) by ~1600x. Both now compute the same thing.
            perceptual_loss = 0.0
            count = 0
            for rf, ff in zip(real_features, fake_features):
                perceptual_loss += tf.reduce_mean(tf.square(rf - ff))
                count += 1
            perceptual_loss /= tf.maximum(tf.cast(count, tf.float32), 1.0)

            total_loss = (pixel_weight * pixel_loss +
                          perceptual_weight * perceptual_loss +
                          kl_weight * kl_loss +
                          INFO_NCE_WEIGHT * info_nce_loss +
                          ADV_LAMBDA * adv_loss)

        g_weights = self.encoder.trainable_weights + self.generator.trainable_weights
        g_grads = g_tape.gradient(total_loss, g_weights)
        g_grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in g_grads]
        self.optimizer.apply_gradients(zip(g_grads, g_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(pixel_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_weight_tracker.update_state(kl_weight)
        self.pixel_weight_tracker.update_state(pixel_weight)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.perceptual_weight_tracker.update_state(perceptual_weight)
        self.disc_loss_tracker.update_state(d_loss)
        self.adv_loss_tracker.update_state(adv_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        z_mean, z_log_var, z = self.encoder([images, labels])
        reconstruction = self.generator([z, labels])
        d_real = self.discriminator([images, labels])
        d_fake = self.discriminator([reconstruction, labels])
        d_loss = (self._bce(tf.ones_like(d_real), d_real)
                  + self._bce(tf.zeros_like(d_fake), d_fake))
        adv_loss = self._bce(tf.ones_like(d_fake), d_fake)

        target = tf.image.resize(images, tf.shape(reconstruction)[1:3])
        pixel_loss = weighted_pixel_loss(target, reconstruction)
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        kl_weight = self.get_kl_weight()
        pixel_weight = self.get_pixel_weight()
        perceptual_weight = self.get_perceptual_weight()
        real_features = self.feature_extractor(images)
        fake_features = self.feature_extractor(reconstruction)
        perceptual_loss = tf.reduce_mean([
            tf.reduce_mean(tf.square(rf - ff))
            for rf, ff in zip(real_features, fake_features)
        ])
        total_loss = (pixel_weight * pixel_loss +
                      perceptual_weight * perceptual_loss +
                      kl_weight * kl_loss +
                      ADV_LAMBDA * adv_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(pixel_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_weight_tracker.update_state(kl_weight)
        self.pixel_weight_tracker.update_state(pixel_weight)
        self.perceptual_loss_tracker.update_state(perceptual_loss)
        self.perceptual_weight_tracker.update_state(perceptual_weight)
        self.disc_loss_tracker.update_state(d_loss)
        self.adv_loss_tracker.update_state(adv_loss)
        return {m.name: m.result() for m in self.metrics}


# ─── Train ────────────────────────────────────────────────────────
class CVAESequence(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batch_size, samples_per_epoch_per_class=None):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.samples_per_epoch_per_class = samples_per_epoch_per_class

        # Handle one-hot labels
        if len(labels.shape) == 2 and labels.shape[1] > 1:
            label_indices = np.argmax(labels, axis=1)
        else:
            label_indices = np.array(labels)

        self.class_indices = {}
        for idx, label in enumerate(label_indices):
            self.class_indices.setdefault(int(label), []).append(idx)
        self.classes = sorted(self.class_indices.keys())
        self.full_indices = np.arange(len(images))
        self._sample_indices()

    def _sample_indices(self):
        if self.samples_per_epoch_per_class is not None and self.samples_per_epoch_per_class > 0:
            sampled_indices = []
            for cls in self.classes:
                cls_idx = np.array(self.class_indices[cls])
                n_sample = min(self.samples_per_epoch_per_class, len(cls_idx))
                sampled = np.random.choice(cls_idx, size=n_sample, replace=False)
                sampled_indices.extend(sampled)
            self.indices = np.array(sampled_indices)
        else:
            self.indices = self.full_indices.copy()
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.images[batch_idx], self.labels[batch_idx]

    def on_epoch_end(self):
        self._sample_indices()


encoder = build_encoder()
generator = build_generator()
discriminator = build_discriminator()
vae = VAEGAN(encoder, generator, discriminator)
vae.compile(g_opt=keras.optimizers.Adam(LR), d_opt=keras.optimizers.Adam(DISC_LR))

train_seq = CVAESequence(x_train, y_train, BATCH_SIZE, samples_per_epoch_per_class=SAMPLES_PER_EPOCH_PER_CLASS)
val_seq = CVAESequence(x_test, y_test, BATCH_SIZE, samples_per_epoch_per_class=None)

# ─── Best Epoch Logger ─────────────────────────────────────────────
class BestEpochLogger(keras.callbacks.Callback):
    def __init__(self, vae_model):
        self.vae = vae_model
        self.best_epoch = 0
        self.best_loss = float('inf')
    def on_epoch_begin(self, epoch, logs=None):
        self.vae.epoch_tracker.assign(epoch)
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_recon_loss')
        if val_loss is not None and val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
    def on_train_end(self, logs=None):
        print(f"\n═══ Best epoch: {self.best_epoch} (val_loss: {self.best_loss:.4f}) ═══")


best_generator_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_generator.h5")

class GeneratorCheckpoint(keras.callbacks.Callback):
    def __init__(self, generator, filepath, monitor='val_recon_loss'):
        super().__init__()
        self.generator = generator
        self.filepath = filepath
        self.monitor = monitor
        self.best_loss = float('inf')
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is not None and current_loss < self.best_loss:
            self.best_loss = current_loss
            self.generator.save_weights(self.filepath)
            print(f"  Saved best generator (val_loss: {current_loss:.4f})")



callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_recon_loss', patience=200, restore_best_weights=True),
    GeneratorCheckpoint(generator, best_generator_path),
    BestEpochLogger(vae),
    keras.callbacks.TensorBoard(log_dir=LOG_NAME, histogram_freq=1),
]

# ─── Phase 1: Train at 14×14 (progressive growing) ────────────────
print(f"\n═══ Phase 1: Training at 14×14 for {PROGRESSIVE_EPOCHS} epochs ═══")
x_train_14 = resize_images(x_train, [14, 14])
x_test_14 = resize_images(x_test, [14, 14])
# Add channel dimension if lost
if len(x_train_14.shape) == 3:
    x_train_14 = x_train_14[..., np.newaxis]
    x_test_14 = x_test_14[..., np.newaxis]

# Build 14×14 model for phase 1
encoder_14 = build_encoder_14()
generator_14 = build_generator_14()

class VAE14(VAE):
    """VAE for 14x14 phase without perceptual loss."""
    def get_lr(self):
        epoch = tf.cast(self.epoch_tracker, tf.float32)
        warmup_epochs = tf.cast(PHA1_LR_WARMUP_EPOCHS, tf.float32)
        decay_epochs = tf.cast(PHA1_LR_DECAY_EPOCHS, tf.float32)
        warmup_progress = epoch / warmup_epochs
        warmup_lr = PHA1_LR_WARMUP_START + (PHA1_LR - PHA1_LR_WARMUP_START) * tf.minimum(1.0, warmup_progress)
        decay_epoch = epoch - warmup_epochs
        decay_progress = tf.minimum(1.0, decay_epoch / tf.maximum(1.0, decay_epochs))
        decay_lr = PHA1_LR + (PHA1_LR_END - PHA1_LR) * decay_progress
        return tf.where(epoch < warmup_epochs, warmup_lr, decay_lr)

    def get_pixel_weight(self):
        """Phase 1 has no pixel schedule. Pixel BCE is the only active term here
        (PHA1_INFO_NCE_WEIGHT is 0 and the KL contributes ~0.1% of it), so
        annealing the weight would just rescale the whole loss -- which is
        indistinguishable from decaying the LR. Constant, and logged as such so
        the metric reports the truth (1.0) instead of an untouched 0.0.
        """
        return tf.constant(PHA1_PIXEL_LOSS_WEIGHT, dtype=tf.float32)

    @property
    def metrics(self):
        # No perceptual loss in phase 1 (VAE14 computes pixel + KL only), so
        # inheriting VAE.metrics would log a permanently-zero perceptual_loss.
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker,
                self.kl_weight_tracker, self.pixel_weight_tracker]

    def train_step(self, data):
        images, labels = data
        self.optimizer.learning_rate = self.get_lr()
        pixel_weight = self.get_pixel_weight()
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([images, labels])
            reconstruction = self.generator([z, labels])
            target = tf.image.resize(images, tf.shape(reconstruction)[1:3])
            pixel_loss = weighted_pixel_loss(target, reconstruction)
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            batch_size = tf.shape(z)[0]
            z_norm = tf.math.l2_normalize(z, axis=1)
            sim_matrix = tf.matmul(z_norm, z_norm, transpose_b=True) / TEMPERATURE
            labels_ce = tf.range(batch_size)
            info_nce_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ce, logits=sim_matrix))
            # Phase 1 KL with configurable warmup
            epoch = tf.cast(self.epoch_tracker, tf.float32)
            kl_warmup_epochs = tf.cast(PHA1_KL_WARMUP_EPOCHS, tf.float32)
            kl_warmup_progress = tf.minimum(1.0, epoch / kl_warmup_epochs)
            kl_weight = PHA1_KL_WARMUP_START + (PHA1_KL_WEIGHT_START - PHA1_KL_WARMUP_START) * kl_warmup_progress
            total_loss = (pixel_weight * pixel_loss +
                          kl_weight * kl_loss +
                          PHA1_INFO_NCE_WEIGHT * info_nce_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(pixel_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_weight_tracker.update_state(kl_weight)
        self.pixel_weight_tracker.update_state(pixel_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        z_mean, z_log_var, z = self.encoder([images, labels])
        reconstruction = self.generator([z, labels])
        target = tf.image.resize(images, tf.shape(reconstruction)[1:3])
        pixel_loss = weighted_pixel_loss(target, reconstruction)
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        pixel_weight = self.get_pixel_weight()
        epoch = tf.cast(self.epoch_tracker, tf.float32)
        kl_warmup_epochs = tf.cast(PHA1_KL_WARMUP_EPOCHS, tf.float32)
        kl_warmup_progress = tf.minimum(1.0, epoch / kl_warmup_epochs)
        kl_weight = PHA1_KL_WARMUP_START + (PHA1_KL_WEIGHT_START - PHA1_KL_WARMUP_START) * kl_warmup_progress
        total_loss = pixel_weight * pixel_loss + kl_weight * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(pixel_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_weight_tracker.update_state(kl_weight)
        self.pixel_weight_tracker.update_state(pixel_weight)
        return {m.name: m.result() for m in self.metrics}

vae_14 = VAE14(encoder_14, generator_14)
vae_14.compile(optimizer=keras.optimizers.Adam(PHA1_LR))
# VAE14 overrides train_step directly (no perceptual loss in phase 1)

train_seq_14 = CVAESequence(x_train_14, y_train, BATCH_SIZE, samples_per_epoch_per_class=SAMPLES_PER_EPOCH_PER_CLASS)
val_seq_14 = CVAESequence(x_test_14, y_test, BATCH_SIZE, samples_per_epoch_per_class=None)

callbacks_14 = [
    keras.callbacks.EarlyStopping(monitor='val_recon_loss', patience=10, restore_best_weights=True),
    BestEpochLogger(vae_14),
    keras.callbacks.TensorBoard(log_dir=LOG_NAME_PHA1, histogram_freq=1),
]

history1 = vae_14.fit(train_seq_14, validation_data=val_seq_14,
                      epochs=PROGRESSIVE_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_14, verbose=2)

# Transfer weights from 14×14 model to 28×28 model
print("\nTransferring weights from 14×14 model to 28×28 model...")

def _dump_weights(model, title):
    print(f"\n=== {title} ===")
    for i, w in enumerate(model.weights):
        print(f"{i:03d} {w.name:<60} {tuple(w.shape)}")
    print("=" * 90)

# one-shot snapshot for exact named mapping later
_dump_weights(encoder_14, "encoder_14 weights")
_dump_weights(encoder, " encoder weights")
_dump_weights(generator_14, "generator_14 weights")
_dump_weights(generator, " generator weights")

def transfer_weights_by_name(src_model, dst_model, name="model"):
    src_by_name = {w.name: w for w in src_model.weights}
    dst_by_name = {w.name: w for w in dst_model.weights}
    transferred = []
    skipped = []
    for w_name, src_w in src_by_name.items():
        if w_name in dst_by_name:
            dst_w = dst_by_name[w_name]
            if tuple(src_w.shape) == tuple(dst_w.shape):
                dst_w.assign(src_w)
                transferred.append(dst_w)
            else:
                skipped.append((w_name, tuple(src_w.shape), tuple(dst_w.shape)))
    print(f"[{name}] transferred {len(transferred)} weights by name; skipped {len(skipped)} due to shape mismatch")
    for w_name, s_shape, d_shape in skipped:
        print(f"  skip {w_name}: {s_shape} -> {d_shape}")
    return transferred

def transfer_weights_by_order(src_model, dst_model, name="model"):
    """Transfer weights by construction order + shape (name-agnostic).

    Walks src/dst weights in parallel by index; copies where shapes match.
    Robust to Keras auto-naming collisions. Used for the generator, whose
    phase-1 and phase-2 builders share the same layer sequence for the
    7x7/14x14 trunk (differing only in the extra 28x28 stage).
    """
    src_w = list(src_model.weights)
    dst_w = list(dst_model.weights)
    transferred = []
    skipped = []
    n = min(len(src_w), len(dst_w))
    for i in range(n):
        sw, dw = src_w[i], dst_w[i]
        if tuple(sw.shape) == tuple(dw.shape):
            dw.assign(sw)
            transferred.append(dw)
        else:
            skipped.append((i, sw.name, tuple(sw.shape), tuple(dw.shape)))
    print(f"[{name}] transferred {len(transferred)}/{n} weights by order; "
          f"skipped {len(skipped)} due to shape mismatch")
    for idx, nm, s_shape, d_shape in skipped:
        print(f"  skip #{idx} {nm}: {s_shape} -> {d_shape}")
    return transferred


enc_transferred = transfer_weights_by_name(encoder_14, encoder, name="encoder")
gen_transferred = transfer_weights_by_name(generator_14, generator, name="generator")
vae.phase1_transferred_weights = enc_transferred + gen_transferred
print(f"Weight transfer complete. {len(vae.phase1_transferred_weights)} weights transferred.")

# ─── Phase 2: Fine-tune at 28×28 ───────────────────────────────────
print(f"\n═══ Phase 2: Fine-tuning at 28×28 for {EPOCHS - PROGRESSIVE_EPOCHS} epochs ═══")
freeze_epochs_end = PROGRESSIVE_EPOCHS + PHA2_TRANSFER_FREEZE_EPOCHS

# Epoch 6..N: transferred weights frozen
print(f"\nPhase 2a: frozen transferred weights for {PHA2_TRANSFER_FREEZE_EPOCHS} epochs")
frozen_weights = list(getattr(vae, 'phase1_transferred_weights', []))
for v in frozen_weights:
    v._trainable = False
vae.compile(g_opt=keras.optimizers.Adam(LR), d_opt=keras.optimizers.Adam(DISC_LR))
history2_frozen = vae.fit(
    train_seq,
    validation_data=val_seq,
    epochs=min(freeze_epochs_end, EPOCHS),
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
    initial_epoch=PROGRESSIVE_EPOCHS,
)

# Unfreeze transferred weights and resume
print("\nUnfreezing transferred weights and resuming Phase 2")
for v in frozen_weights:
    v._trainable = True
vae.compile(g_opt=keras.optimizers.Adam(LR), d_opt=keras.optimizers.Adam(DISC_LR))
history2 = vae.fit(
    train_seq,
    validation_data=val_seq,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
    initial_epoch=freeze_epochs_end,
)

# ─── Load best generator weights (in case EarlyStopping didn't trigger) ──
if os.path.exists(best_generator_path):
    generator.load_weights(best_generator_path)
    print(f"\nLoaded best generator weights from {best_generator_path}")

# ─── Save ─────────────────────────────────────────────────────────
save_dir = os.path.dirname(os.path.abspath(__file__))
generator.save(os.path.join(save_dir, "cvae_generator.h5"))
print(f"\nGenerator saved to {save_dir}/cvae_generator.h5")
