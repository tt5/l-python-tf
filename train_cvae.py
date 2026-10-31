#!/usr/bin/env python3
"""train_cvae.py

VAEGAN for synthetic data.
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
EPOCHS = 230
BATCH_SIZE = 512
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

LR = 0.003
LR_WARMUP_START = 0.00005
LR_WARMUP_EPOCHS = 5
LR_DECAY_EPOCHS = EPOCHS - LR_WARMUP_EPOCHS
LR_END = 0.00005
INFO_NCE_WEIGHT = 0.001
TEMPERATURE = 0.9
PIXEL_POS_WEIGHT = 1.0
DROPOUT = 0.3
SPATIAL_DROPOUT = 0.0
L2_REG = 0.00001
# VAE-GAN hybrid: adversarial term + discriminator LR
ADV_LAMBDA = 0.5
ADV_WARMUP_EPOCHS = 10
DISC_LR = 0.0001
D_BATCH_INTERVAL = 1  # update discriminator every N batches (G updates every batch)
D_REAL_LABEL = 1.0    # one-sided label smoothing for the real class

KL_WARMUP_START = 0.0001
KL_WEIGHT_START = 0.014
KL_WEIGHT_TARGET = 0.003
KL_WARMUP_EPOCHS = 55
KL_DECAY_EPOCHS = 80

PIXEL_WEIGHT_START = 1.1 * 16
PIXEL_WEIGHT_TARGET = 0.5 * 16
PIXEL_DECAY_EPOCHS = EPOCHS - 1

PERC_WARMUP_START = 0.01
PERC_WEIGHT_START = 4000.0
PERC_WEIGHT_TARGET = 500.0
PERC_WARMUP_EPOCHS = 80
PERC_DECAY_EPOCHS = EPOCHS - PERC_WARMUP_EPOCHS - 1
LOG_NAME = 'logs/gen1'

def weighted_pixel_loss(target, reconstruction):
    """Foreground-weighted pixel BCE, as a PER-PIXEL MEAN.

    keras binary_crossentropy returns shape [B,H,W] (the channel axis is
    squeezed), while `target` is [B,H,W,1] -- so the weights are squeezed
    to match before multiplying.
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
    # x * gate) so it can be fused with FiLM by CollabModulation at c1
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

    b3 = layers.Conv2D(16, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc28_conv3")(img)
    b5 = layers.Conv2D(16, 5, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc28_conv5")(img)
    b7 = layers.Conv2D(16, 7, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc28_conv7")(img)
    x28 = layers.Concatenate(name="enc28_cat")([b3, b5, b7])   # 3 x ch @ 28x28
    x28 = layers.LayerNormalization(name="enc28_ln")(x28)
    sc28 = layers.Conv2D(48, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc28_proj")(img)
    x28 = layers.Add()([x28, sc28])
    x28 = layers.Activation("relu")(x28)

    att = spatial_attention_map(x28, lbl, k=5, name="enc28_spatial_attn")
    gamma28 = layers.Reshape((1, 1, 48))(2.0 * layers.Dense(48, activation="sigmoid", name="enc28_film_g")(lbl))
    beta28 = layers.Reshape((1, 1, 48))(layers.Dense(48, name="enc28_film_b")(lbl))
    x28 = CollabModulation(name="enc28_collab")([x28, gamma28, att, beta28])
    #c0 = x28

    c0 = layers.Conv2D(48, 3, strides=1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv0")(x28)
    c0 = tf.keras.layers.LayerNormalization(name='enc_conv0_ln')(c0)
    c0 = layers.Activation("relu")(c0)
    skip0 = layers.Conv2D(48, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv0_skip")(x28)
    skip0 = tf.keras.layers.LayerNormalization(name='enc_conv0_skip_ln')(skip0)
    c0 = layers.Add()([c0, skip0])
    c0 = layers.Activation("relu")(c0)

    # Conv block 1: 28x28 → 14x14
    c1 = layers.Conv2D(96, 3, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv1")(c0)
    c1 = tf.keras.layers.LayerNormalization(name='enc_14_ln')(c1)
    sc = layers.Conv2D(96, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc28_skip_proj")(layers.AveragePooling2D(2)(c0))
    c1 = layers.Add()([c1, sc])
    c1 = layers.Activation("relu")(c1)

    att = spatial_attention_map(c1, lbl, k=5, name="enc_spatial_attn")
    gamma14 = layers.Reshape((1, 1, 96))(2.0 * layers.Dense(96, activation="sigmoid", name="enc_film14_g")(lbl))
    beta14 = layers.Reshape((1, 1, 96))(layers.Dense(96, name="enc_film14_b")(lbl))
    c1 = CollabModulation(name="enc_collab14")([c1, gamma14, att, beta14])

    c1b = layers.Conv2D(96, 3, strides=1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv1b")(c1)
    c1b = tf.keras.layers.LayerNormalization(name='enc_conv1b_ln')(c1b)
    c1b = layers.Activation("relu")(c1b)
    skip1b = layers.Conv2D(96, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv1b_skip")(c1)
    skip1b = tf.keras.layers.LayerNormalization(name='enc_conv1b_skip_ln')(skip1b)
    c1b = layers.Add()([c1b, skip1b])
    c1 = layers.Activation("relu")(c1b)

    # Conv block 2: 14x14 → 7x7
    c2 = layers.Conv2D(192, 3, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv2")(c1)
    c2 = tf.keras.layers.LayerNormalization(name='enc_conv2_ln')(c2)
    c2 = layers.Activation("relu")(c2)
    skip = layers.Conv2D(192, 1, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv2_skip")(c1)
    skip = tf.keras.layers.LayerNormalization(name='enc_conv2_skip_ln')(skip)
    c2 = layers.Add()([c2, skip])
    c2 = layers.Activation("relu")(c2)
    #c3 = c2

    # Conv block 3: 7x7 -> 7x7
    c3 = layers.Conv2D(192, 3, strides=1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv3")(c2)
    c3 = tf.keras.layers.LayerNormalization(name='enc_conv3_ln')(c3)
    c3 = layers.Activation("relu")(c3)
    skip2 = layers.Conv2D(192, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv3_skip")(c2)
    skip2 = tf.keras.layers.LayerNormalization(name='enc_conv3_skip_ln')(skip2)
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
    main = layers.Dense(768, activation="relu", name="enc28_head_dense", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(main)
    main = layers.LayerNormalization(name="enc28_head_ln")(main)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(main)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(main)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model([img, lbl], [z_mean, z_log_var, z], name="encoder")


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


# ─── Generator with FiLM Conditioning ──────────────────────────────
def build_generator():
    z = keras.Input(shape=(LATENT_DIM,), name="latent_input")
    lbl = keras.Input(shape=(NUM_CLASSES,), name="label_input")

    def make_film(channels, tag):
        def film_fn(label):
            gamma = 2.0 * layers.Dense(channels, activation="sigmoid", name=f"{tag}_g")(label)
            beta = layers.Dense(channels, name=f"{tag}_b")(label)
            return gamma, beta
        return film_fn

    # Label-conditioned spatial gate (CBAM-style). Channel-pool (avg+max) +
    # broadcast label -> kxk conv -> sigmoid gate. Returns the GATE (not
    # x * gate) so it can be fused with FiLM by CollabModulation at c1
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

    # Project latent to 1x1x1024, then double resolution each stage.
    x = layers.Dense(1 * 1 * 512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_proj0")(z)
    x = layers.Reshape((1, 1, 512))(x)

    # 1x1 -> 2x2
    x = layers.Conv2DTranspose(256, 4, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_up1")(x)
    x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)
    x = tf.keras.layers.Activation("relu")(x)

    x2 = layers.Dense(2 * 2 * 256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_proj1")(z)
    x2 = layers.Reshape((2, 2, 256))(x2)

    x = tf.keras.layers.Add()([
        x2,
        tf.keras.layers.Conv2D(256, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x),
    ])

    # 2x2 -> 4x4
    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_up2")(x)
    x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)
    x = tf.keras.layers.Activation("relu")(x)

    x2 = layers.Dense(4 * 4 * 128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_proj2")(z)
    x2 = layers.Reshape((4, 4, 128))(x2)

    x = tf.keras.layers.Add()([
        x2,
        tf.keras.layers.Conv2D(128, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x),
    ])

    # 4x4 -> 8x8
    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_up3")(x)
    x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)
    x = tf.keras.layers.Activation("relu")(x)

    x2 = layers.Dense(8 * 8 * 64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_proj3")(z)
    x2 = layers.Reshape((8, 8, 64))(x2)

    x = tf.keras.layers.Add()([
        x2,
        tf.keras.layers.Conv2D(64, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x),
    ])

    att = spatial_attention_map(x, lbl, k=5, name="gen8_spatial_attn")
    gamma = layers.Reshape((1, 1, 64))(2.0 * layers.Dense(64, activation="sigmoid", name="gen8_film_g")(lbl))
    beta = layers.Reshape((1, 1, 64))(layers.Dense(64, name="gen8_film_b")(lbl))
    x = CollabModulation(name="gen_collab8")([x, gamma, att, beta])

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)

    # 8x8 -> 16x16
    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_up4")(x)
    x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)
    x = tf.keras.layers.Activation("relu")(x)

    att = spatial_attention_map(x, lbl, k=5, name="gen16_spatial_attn")
    gamma = layers.Reshape((1, 1, 32))(2.0 * layers.Dense(32, activation="sigmoid", name="gen16_film_g")(lbl))
    beta = layers.Reshape((1, 1, 32))(layers.Dense(32, name="gen16_film_b")(lbl))
    x = CollabModulation(name="gen_collab16")([x, gamma, att, beta])

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)

    # 16x16 -> 32x32
    x = layers.Conv2DTranspose(16, 4, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="gen_up5")(x)
    x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Crop 32x32 -> 28x28
    x = layers.Cropping2D(cropping=((2, 2), (2, 2)))(x)

    att = spatial_attention_map(x, lbl, k=5, name="gen28_spatial_attn")
    gamma = layers.Reshape((1, 1, 16))(2.0 * layers.Dense(16, activation="sigmoid", name="gen28_film_g")(lbl))
    beta = layers.Reshape((1, 1, 16))(layers.Dense(16, name="gen28_film_b")(lbl))
    x = CollabModulation(name="gen_collab28")([x, gamma, att, beta])

    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = tf.keras.layers.LayerNormalization(center=True, scale=True)(x)

    # Final refinement conv -> sigmoid
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
        kl_epoch = epoch - tf.cast(0, tf.float32)
        warmup_epochs = tf.cast(KL_WARMUP_EPOCHS, tf.float32)
        decay_epochs = tf.cast(KL_DECAY_EPOCHS, tf.float32)
        start = tf.constant(KL_WARMUP_START, dtype=tf.float32)
        mid = tf.constant(KL_WEIGHT_START, dtype=tf.float32)
        end = tf.constant(KL_WEIGHT_TARGET, dtype=tf.float32)
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
        epoch = tf.cast(self.epoch_tracker, tf.float32)
        p2_epoch = epoch - tf.cast(0, tf.float32)
        start = tf.constant(PIXEL_WEIGHT_START, dtype=tf.float32)
        end = tf.constant(PIXEL_WEIGHT_TARGET, dtype=tf.float32)
        progress = tf.clip_by_value(
            p2_epoch / tf.cast(PIXEL_DECAY_EPOCHS, tf.float32), 0.0, 1.0)
        return start + (end - start) * progress

    def get_perceptual_weight(self):
        epoch = tf.cast(self.epoch_tracker, tf.float32)
        p2_epoch = epoch - tf.cast(0, tf.float32)
        warmup_epochs = tf.cast(PERC_WARMUP_EPOCHS, tf.float32)
        decay_epochs = tf.cast(PERC_DECAY_EPOCHS, tf.float32)
        start = tf.constant(PERC_WARMUP_START, dtype=tf.float32)
        mid = tf.constant(PERC_WEIGHT_START, dtype=tf.float32)
        end = tf.constant(PERC_WEIGHT_TARGET, dtype=tf.float32)
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


class VAEGAN(VAE):
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

    def get_adv_weight(self):
        epoch = tf.cast(self.epoch_tracker, tf.float32)
        warmup_epochs = tf.cast(ADV_WARMUP_EPOCHS, tf.float32)
        progress = tf.clip_by_value(epoch / warmup_epochs, 0.0, 1.0)
        return tf.constant(ADV_LAMBDA, dtype=tf.float32) * progress

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
                          self.get_adv_weight() * adv_loss)

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
                      self.get_adv_weight() * adv_loss)
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

vae.compile(g_opt=keras.optimizers.Adam(LR), d_opt=keras.optimizers.Adam(DISC_LR))
history2 = vae.fit(
    train_seq,
    validation_data=val_seq,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
    initial_epoch=0,
)

# ─── Load best generator weights (in case EarlyStopping didn't trigger) ──
if os.path.exists(best_generator_path):
    generator.load_weights(best_generator_path)
    print(f"\nLoaded best generator weights from {best_generator_path}")

# ─── Save ─────────────────────────────────────────────────────────
save_dir = os.path.dirname(os.path.abspath(__file__))
generator.save(os.path.join(save_dir, "cvae_generator.h5"))
print(f"\nGenerator saved to {save_dir}/cvae_generator.h5")
