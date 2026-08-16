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

from utils.mnist_loader import load_data

# ─── Config ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic" / "data"
LATENT_DIM = 256
PHA1_LATENT_DIM = 64
EPOCHS = 90
BATCH_SIZE = 128
NUM_CLASSES = 5

parser = argparse.ArgumentParser()
parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--max-samples-per-class", type=int, default=None, help="Maximum samples per class")
args = parser.parse_args()

NUM_CLASSES = args.num_classes
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
MAX_SAMPLES_PER_CLASS = args.max_samples_per_class

PROGRESSIVE_EPOCHS = 3 # Phase 1: 14x14
LR = 0.001
LR_WARMUP_START = 0.000005
LR_WARMUP_EPOCHS = 2
LR_DECAY_EPOCHS = EPOCHS - LR_WARMUP_EPOCHS
LR_END = 0.0001
PIXEL_LOSS_WEIGHT = 0.8
PERCEPTUAL_LOSS_WEIGHT = 0.2
INFO_NCE_WEIGHT = 0.0
TEMPERATURE = 1.0
DROPOUT = 0.35
SPATIAL_DROPOUT = 0.35
L2_REG = 0.00006
# Phase 2 KL schedule (configurable start)
PHA2_KL_WARMUP_START = 0.0003
PHA2_KL_WEIGHT_START = 0.5
PHA2_KL_WEIGHT_TARGET = 0.2
PHA2_KL_WARMUP_EPOCHS = 5
PHA2_KL_DECAY_EPOCHS = EPOCHS
PHA2_TRANSFER_FREEZE_EPOCHS = 1
# Phase 1 (14x14) specific config
PHA1_LR = 0.0003
PHA1_LR_WARMUP_START = 0.0000001
PHA1_LR_WARMUP_EPOCHS = 2
PHA1_LR_END = 0.00005
PHA1_LR_DECAY_EPOCHS = PROGRESSIVE_EPOCHS - PHA1_LR_WARMUP_EPOCHS
PHA1_PIXEL_LOSS_WEIGHT = 1.0
PHA1_KL_WARMUP_EPOCHS = 1
PHA1_KL_WARMUP_START = 0.0000001
PHA1_KL_WEIGHT_START = 0.01
PHA1_INFO_NCE_WEIGHT = 0.0
LOG_NAME = 'logs/gen1'
LOG_NAME_PHA1 = 'logs/gen1/phase1'

# ─── Data ──────────────────────────────────────────────────────────
def load_data_cvae():
    (x_train, y_train), (x_test, y_test) = load_data(DATA_DIR, NUM_CLASSES, flip_pixels=7*28)
    if MAX_SAMPLES_PER_CLASS is not None and MAX_SAMPLES_PER_CLASS > 0:
        mask_train = np.zeros(len(x_train), dtype=bool)
        mask_test = np.zeros(len(x_test), dtype=bool)
        for c in range(NUM_CLASSES - 1):
            idx_tr = np.where(y_train == c)[0]
            idx_te = np.where(y_test == c)[0]
            take_tr = min(MAX_SAMPLES_PER_CLASS, len(idx_tr))
            take_te = min(MAX_SAMPLES_PER_CLASS, len(idx_te))
            mask_train[idx_tr[:take_tr]] = True
            mask_test[idx_te[:take_te]] = True
        x_train, y_train = x_train[mask_train], y_train[mask_train]
        x_test, y_test = x_test[mask_test], y_test[mask_test]
        train_perm = np.random.permutation(len(x_train))
        test_perm = np.random.permutation(len(x_test))
        x_train, y_train = x_train[train_perm], y_train[train_perm]
        x_test, y_test = x_test[test_perm], y_test[test_perm]
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    print(f"Using {len(x_train)} train / {len(x_test)} test samples")
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

    # Conv block 1: 28x28 → 14x14
    c1 = layers.Conv2D(64, 3, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv1")(img)
    c1 = layers.Activation("relu")(c1)
    # SE attention 1
    se1 = layers.GlobalAveragePooling2D()(c1)
    se1 = layers.Dense(64 // 4, activation="relu")(se1)
    se1 = layers.Dense(64, activation="sigmoid")(se1)
    se1 = layers.Reshape((1, 1, 64))(se1)
    c1 = layers.Multiply()([c1, se1])
    c1 = layers.SpatialDropout2D(SPATIAL_DROPOUT)(c1)

    # Conv block 2: 14x14 → 7x7
    c2 = layers.Conv2D(128, 3, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv2")(c1)
    c2 = layers.Activation("relu")(c2)
    # Residual connection: project c1 (14x14x64) to match c2 (7x7x128)
    skip = layers.Conv2D(128, 1, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv2_skip")(c1)
    c2 = layers.Add()([c2, skip])
    c2 = layers.Activation("relu")(c2)
    # SE attention 2
    se2 = layers.GlobalAveragePooling2D()(c2)
    se2 = layers.Dense(128 // 4, activation="relu")(se2)
    se2 = layers.Dense(128, activation="sigmoid")(se2)
    se2 = layers.Reshape((1, 1, 128))(se2)
    c2 = layers.Multiply()([c2, se2])
    c2 = layers.SpatialDropout2D(SPATIAL_DROPOUT)(c2)

    # Cross-scale shortcut from c1 into block 3
    pooled_c1 = layers.AveragePooling2D(2)(c1)
    c2 = layers.Concatenate()([c2, pooled_c1])

    # Conv block 3: 7x7 → 7x7 (stride 1)
    c3 = layers.Conv2D(256, 3, strides=1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(c2)
    c3 = layers.Activation("relu")(c3)
    # Residual connection: project c2 (7x7x128) to match c3 (7x7x256)
    skip2 = layers.Conv2D(256, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(c2)
    c3 = layers.Add()([c3, skip2])
    c3 = layers.Activation("relu")(c3)
    # SE attention 3
    se3 = layers.GlobalAveragePooling2D()(c3)
    se3 = layers.Dense(256 // 4, activation="relu")(se3)
    se3 = layers.Dense(256, activation="sigmoid")(se3)
    se3 = layers.Reshape((1, 1, 256))(se3)
    c3_pre = c3
    c3 = layers.Multiply()([c3, se3])
    c3 = layers.Activation("relu")(c3)
    skip3 = layers.Conv2D(256, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(c3_pre)
    c3 = layers.Add()([c3, skip3])

    # Bottleneck
    x = layers.Flatten()(c3)
    x = layers.Concatenate()([x, lbl])
    x = layers.LayerNormalization()(x)
    x = layers.Dense(640, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = layers.Dropout(DROPOUT)(x)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model([img, lbl], [z_mean, z_log_var, z], name="encoder")


# ─── Encoder for 14×14 Phase 1 ─────────────────────────────────────
def build_encoder_14():
    img = keras.Input(shape=(14, 14, 1), name="image_input")
    lbl = keras.Input(shape=(NUM_CLASSES,), name="label_input")

    # Block 1: 14x14 -> 7x7, mirrored from 28x28 encoder
    c1 = layers.Conv2D(64, 3, strides=2, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv1")(img)
    c1 = layers.Activation("relu")(c1)
    se1 = layers.GlobalAveragePooling2D()(c1)
    se1 = layers.Dense(64 // 4, activation="relu", name="dense")(se1)
    se1 = layers.Dense(64, activation="sigmoid", name="dense_1")(se1)
    se1 = layers.Reshape((1, 1, 64))(se1)
    c1 = layers.Multiply()([c1, se1])
    c1 = layers.SpatialDropout2D(SPATIAL_DROPOUT)(c1)

    # Block 2: 7x7 -> 7x7, mirrored from 14x14 encoder
    c2 = layers.Conv2D(128, 3, strides=1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv2")(c1)
    c2 = layers.Activation("relu")(c2)
    skip = layers.Conv2D(128, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv2_skip")(c1)
    c2 = layers.Add()([c2, skip])
    c2 = layers.Activation("relu")(c2)
    se2 = layers.GlobalAveragePooling2D()(c2)
    se2 = layers.Dense(128 // 4, activation="relu", name="dense_2")(se2)
    se2 = layers.Dense(128, activation="sigmoid", name="dense_3")(se2)
    se2 = layers.Reshape((1, 1, 128))(se2)
    c2 = layers.Multiply()([c2, se2])
    c2 = layers.SpatialDropout2D(SPATIAL_DROPOUT)(c2)

    # Bottleneck
    x = layers.Flatten()(c2)
    x = layers.Concatenate()([x, lbl])
    x = layers.LayerNormalization()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="dense_6")(x)
    x = layers.Dropout(DROPOUT)(x)
    z_mean = layers.Dense(PHA1_LATENT_DIM, name="z_mean", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    z_log_var = layers.Dense(PHA1_LATENT_DIM, name="z_log_var", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model([img, lbl], [z_mean, z_log_var, z], name="encoder_14")


# ─── Upsample via Sub-pixel Convolution ────────────────────────────
def upsample(x, target_size):
    """Upsample using sub-pixel convolution (depth_to_space)."""
    channels = x.shape[-1]
    x = layers.Conv2D(channels * 4, 3, padding='same', activation='relu')(x)
    x = tf.nn.depth_to_space(x, 2)
    return x


# ─── Generator for 14×14 Phase 1 ────────────────────────────────────
def build_generator_14():
    z = keras.Input(shape=(PHA1_LATENT_DIM,), name="latent_input")
    lbl = keras.Input(shape=(NUM_CLASSES,), name="label_input")

    def make_film(channels):
        def film_fn(label):
            gamma = layers.Dense(channels, activation="sigmoid")(label)
            beta = layers.Dense(channels)(label)
            return gamma, beta
        return film_fn

    # Initial projection to 7x7, mirrored from 28x28 generator
    x = layers.Dense(7 * 7 * 128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="dense_7")(z)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.Reshape((7, 7, 128))(x)
    gamma7, beta7 = make_film(128)(lbl)
    gamma7 = layers.Reshape((1, 1, 128), name="reshape")(gamma7)
    beta7 = layers.Reshape((1, 1, 128), name="reshape_1")(beta7)
    x = layers.Multiply()([x, gamma7])
    x = layers.Add()([x, beta7])
    se_init = layers.GlobalAveragePooling2D()(x)
    se_init = layers.Dense(128 // 4, activation="relu", name="dense_8")(se_init)
    se_init = layers.Dense(128, activation="sigmoid", name="dense_9")(se_init)
    se_init = layers.Reshape((1, 1, 128))(se_init)
    x = layers.Multiply()([x, se_init])
    sa_init = layers.Conv2D(1, 1, activation='sigmoid', name="conv2d_3")(x)
    x = layers.Multiply()([x, sa_init])
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT)(x)

    # 7x7 block with FiLM, mirrored from 28x28 generator
    x = layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="conv2d_4")(x)
    gamma7c, beta7c = make_film(128)(lbl)
    gamma7c = layers.Reshape((1, 1, 128), name="reshape_2")(gamma7c)
    beta7c = layers.Reshape((1, 1, 128), name="reshape_3")(beta7c)
    x = layers.Multiply()([x, gamma7c])
    x = layers.Add()([x, beta7c])
    se7 = layers.GlobalAveragePooling2D()(x)
    se7 = layers.Dense(128 // 4, activation="relu", name="dense_10")(se7)
    se7 = layers.Dense(128, activation="sigmoid", name="dense_11")(se7)
    se7 = layers.Reshape((1, 1, 128))(se7)
    x = layers.Multiply()([x, se7])
    sa7 = layers.Conv2D(1, 1, activation='sigmoid', name="conv2d_5")(x)
    x = layers.Multiply()([x, sa7])
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT)(x)
    skip_7 = layers.Conv2D(128, 1, padding="same", name="conv2d_6")(x)

    # 7x7 → 14x14, mirrored from 28x28 generator
    x = upsample(x, [14, 14])
    x = layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="conv2d_7")(x)
    gamma14, beta14 = make_film(128)(lbl)
    gamma14 = layers.Reshape((1, 1, 128), name="reshape_4")(gamma14)
    beta14 = layers.Reshape((1, 1, 128), name="reshape_5")(beta14)
    x = layers.Multiply()([x, gamma14])
    x = layers.Add()([x, beta14])
    sa14_pre = layers.Conv2D(1, 1, activation='sigmoid', name="conv2d_8")(x)
    x = layers.Multiply()([x, sa14_pre])
    skip_7_up = upsample(skip_7, [14, 14])
    x = layers.Add()([x, skip_7_up])
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(1, 3, padding="same", activation="sigmoid", dtype="float32", name="conv2d_9")(x)
    return keras.Model([z, lbl], x, name="generator_14")


# ─── Generator with FiLM Conditioning ──────────────────────────────
def build_generator():
    z = keras.Input(shape=(LATENT_DIM,), name="latent_input")
    lbl = keras.Input(shape=(NUM_CLASSES,), name="label_input")

    # FiLM helper: produce scale (gamma) and shift (beta) from label
    def make_film(channels):
        def film_fn(label):
            gamma = layers.Dense(channels, activation="sigmoid")(label)  # [0, 1] range
            beta = layers.Dense(channels)(label)
            return gamma, beta
        return film_fn

    # Initial projection
    x = layers.Dense(7 * 7 * 128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(z)
    x = layers.Reshape((7, 7, 128))(x)
    # FiLM at 7x7
    gamma7, beta7 = make_film(128)(lbl)
    gamma7 = layers.Reshape((1, 1, 128))(gamma7)
    beta7 = layers.Reshape((1, 1, 128))(beta7)
    x = layers.Multiply()([x, gamma7])
    x = layers.Add()([x, beta7])
    # SE attention on initial projection
    se_init = layers.GlobalAveragePooling2D()(x)
    se_init = layers.Dense(128 // 4, activation="relu")(se_init)
    se_init = layers.Dense(128, activation="sigmoid")(se_init)
    se_init = layers.Reshape((1, 1, 128))(se_init)
    x = layers.Multiply()([x, se_init])
    # Spatial attention before conv
    sa_init = layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, sa_init])
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT)(x)

    # 7x7 → 14x14
    x = layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    # FiLM at 7x7 (after conv)
    gamma7c, beta7c = make_film(128)(lbl)
    gamma7c = layers.Reshape((1, 1, 128))(gamma7c)
    beta7c = layers.Reshape((1, 1, 128))(beta7c)
    x = layers.Multiply()([x, gamma7c])
    x = layers.Add()([x, beta7c])
    # SE attention 7x7
    se7 = layers.GlobalAveragePooling2D()(x)
    se7 = layers.Dense(128 // 4, activation="relu")(se7)
    se7 = layers.Dense(128, activation="sigmoid")(se7)
    se7 = layers.Reshape((1, 1, 128))(se7)
    x = layers.Multiply()([x, se7])
    # Spatial attention 7x7
    sa7 = layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, sa7])
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT)(x)
    skip_7 = layers.Conv2D(128, 1, padding="same")(x)
    x = upsample(x, [14, 14])
    x = layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    # FiLM at 14x14
    gamma14, beta14 = make_film(128)(lbl)
    gamma14 = layers.Reshape((1, 1, 128))(gamma14)
    beta14 = layers.Reshape((1, 1, 128))(beta14)
    x = layers.Multiply()([x, gamma14])
    x = layers.Add()([x, beta14])
    # Spatial attention before residual
    sa14_pre = layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, sa14_pre])
    # Residual: project skip_7 to 14x14 and add
    skip_7_up = upsample(skip_7, [14, 14])
    x = layers.Add()([x, skip_7_up])
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT)(x)

    # 14x14 → 28x28
    x = layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    # FiLM at 28x28
    gamma28, beta28 = make_film(64)(lbl)
    gamma28 = layers.Reshape((1, 1, 64))(gamma28)
    beta28 = layers.Reshape((1, 1, 64))(beta28)
    x = layers.Multiply()([x, gamma28])
    x = layers.Add()([x, beta28])
    # SE attention 14x14
    se14 = layers.GlobalAveragePooling2D()(x)
    se14 = layers.Dense(64 // 4, activation="relu")(se14)
    se14 = layers.Dense(64, activation="sigmoid")(se14)
    se14 = layers.Reshape((1, 1, 64))(se14)
    x = layers.Multiply()([x, se14])
    # Spatial attention 14x14
    sa14 = layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, sa14])
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT)(x)
    skip_14 = layers.Conv2D(64, 1, padding="same")(x)
    x = upsample(x, [28, 28])
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.SpatialDropout2D(SPATIAL_DROPOUT)(x)
    skip_14_up = upsample(skip_14, [28, 28])
    x = layers.Add()([x, skip_14_up])
    x = layers.Activation("relu")(x)
    x = layers.Dropout(DROPOUT)(x)

    x = layers.Conv2D(1, 3, padding="same", activation="sigmoid", dtype="float32")(x)
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

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker, self.kl_weight_tracker]

    def train_step(self, data):
        images, labels = data

        # Update LR with warmup
        self.optimizer.learning_rate = self.get_lr()

        # Get KL weight
        kl_weight = self.get_kl_weight()

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([images, labels])
            reconstruction = self.generator([z, labels])

            # Pixel-level reconstruction loss (BCE)
            # Resize target to match reconstruction size (no-op if same size)
            target = tf.image.resize(images, tf.shape(reconstruction)[1:3])
            pixel_loss = tf.reduce_mean(tf.reduce_sum(
                keras.losses.binary_crossentropy(target, reconstruction), axis=(1, 2)))

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
                diff = tf.reduce_mean(tf.square(rf - ff))
                n = tf.cast(tf.size(rf), tf.float32)
                perceptual_loss += diff / n
                count += 1
            perceptual_loss /= tf.maximum(tf.cast(count, tf.float32), 1.0)

            # Combined loss
            total_loss = (PIXEL_LOSS_WEIGHT * pixel_loss +
                          PERCEPTUAL_LOSS_WEIGHT * perceptual_loss +
                          kl_weight * kl_loss +
                          INFO_NCE_WEIGHT * info_nce_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(pixel_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_weight_tracker.update_state(kl_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        z_mean, z_log_var, z = self.encoder([images, labels])
        reconstruction = self.generator([z, labels])
        target = tf.image.resize(images, tf.shape(reconstruction)[1:3])
        pixel_loss = tf.reduce_mean(tf.reduce_sum(
            keras.losses.binary_crossentropy(target, reconstruction), axis=(1, 2)))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        kl_weight = self.get_kl_weight()
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
        total_loss = (PIXEL_LOSS_WEIGHT * pixel_loss +
                      PERCEPTUAL_LOSS_WEIGHT * perceptual_loss +
                      kl_weight * kl_loss +
                      INFO_NCE_WEIGHT * info_nce_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(pixel_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.kl_weight_tracker.update_state(kl_weight)
        return {m.name: m.result() for m in self.metrics}


# ─── Train ────────────────────────────────────────────────────────
encoder = build_encoder()
generator = build_generator()
vae = VAE(encoder, generator)
vae.compile(optimizer=keras.optimizers.Adam(LR))

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
    keras.callbacks.EarlyStopping(monitor='val_recon_loss', patience=90, restore_best_weights=True),
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

    def train_step_14(self, data):
        images, labels = data
        self.optimizer.learning_rate = self.get_lr()
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([images, labels])
            reconstruction = self.generator([z, labels])
            target = tf.image.resize(images, tf.shape(reconstruction)[1:3])
            pixel_loss = tf.reduce_mean(tf.reduce_sum(
                keras.losses.binary_crossentropy(target, reconstruction), axis=(1, 2)))
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
            total_loss = (PHA1_PIXEL_LOSS_WEIGHT * pixel_loss +
                          kl_weight * kl_loss +
                          PHA1_INFO_NCE_WEIGHT * info_nce_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(pixel_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        z_mean, z_log_var, z = self.encoder([images, labels])
        reconstruction = self.generator([z, labels])
        target = tf.image.resize(images, tf.shape(reconstruction)[1:3])
        pixel_loss = tf.reduce_mean(tf.reduce_sum(
            keras.losses.binary_crossentropy(target, reconstruction), axis=(1, 2)))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        epoch = tf.cast(self.epoch_tracker, tf.float32)
        kl_warmup_epochs = tf.cast(PHA1_KL_WARMUP_EPOCHS, tf.float32)
        kl_warmup_progress = tf.minimum(1.0, epoch / kl_warmup_epochs)
        kl_weight = PHA1_KL_WARMUP_START + (PHA1_KL_WEIGHT_START - PHA1_KL_WARMUP_START) * kl_warmup_progress
        total_loss = PHA1_PIXEL_LOSS_WEIGHT * pixel_loss + kl_weight * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(pixel_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

vae_14 = VAE14(encoder_14, generator_14)
vae_14.compile(optimizer=keras.optimizers.Adam(PHA1_LR))
# Use train_step_14 which skips perceptual loss
vae_14.train_step = vae_14.train_step_14

callbacks_14 = [
    keras.callbacks.EarlyStopping(monitor='val_recon_loss', patience=10, restore_best_weights=True),
    BestEpochLogger(vae_14),
    keras.callbacks.TensorBoard(log_dir=LOG_NAME_PHA1, histogram_freq=1),
]

history1 = vae_14.fit(x_train_14, y_train, validation_data=(x_test_14, y_test),
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
vae.compile(optimizer=keras.optimizers.Adam(LR))
history2_frozen = vae.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
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
vae.compile(optimizer=keras.optimizers.Adam(LR))
history2 = vae.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
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
