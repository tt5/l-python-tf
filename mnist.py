#!/usr/bin/env python3
"""mnist.py

Train a classifier on synthetic pictogram data.
Loads data from synthetic dataset (classes 0-10).

Usage:
    /home/n/miniconda3/envs/tf/bin/python mnist.py
"""

import tensorflow as tf
import numpy as np
import sys
import argparse
from pathlib import Path

from utils.mnist_loader import load_data

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic" / "data"
NUM_CLASSES = 5
EPOCHS = 35
BATCH_SIZE = 128
BASE_LR = 0.00005
GRAD_ADAPTIVE_MIN_LR = 0.0000014
GRAD_ADAPTIVE_MAX_LR = 0.000086
GRAD_ADAPTIVE_FACTOR = 0.0000045
GRAD_ADAPTIVE_BATCH_INTERVAL = 3
GRAD_NORM_EMA_DECAY = 0.93
GRAD_NORM_TARGET = 1.0
DROPOUT_START = 0.45
DROPOUT_END = 0.25
DROPOUT = DROPOUT_START
SPATIAL_DROPOUT = 0.2
DENSE_SIZE = 640
L2_DECAY = 0.00001
GCE_Q = 0.3
EARLY_STOP_PATIENCE = 10
GAUSSIAN_NOISE = 0.00006
GAUSSIAN_NOISE_DECAY_EPOCHS = 20
GAUSSIAN_NOISE_END = 0.0
LOG_DIR = "logs/cls1b"


AUGMENT = True  # On-the-fly augmentation: random rotate/transpose


parser = argparse.ArgumentParser()
parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--max-samples-per-class", type=int, default=None, help="Maximum samples per class")
parser.add_argument("--samples-per-epoch-per-class", type=int, default=None, help="Random subset size per class each epoch")
parser.add_argument("--pbm-dir", type=str, default=None, help="Load P4 PBM images from this directory instead of .npy")
args = parser.parse_args()

NUM_CLASSES = args.num_classes
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
MAX_SAMPLES_PER_CLASS = args.max_samples_per_class
SAMPLES_PER_EPOCH_PER_CLASS = args.samples_per_epoch_per_class
PBM_DIR = Path(args.pbm_dir) if args.pbm_dir else None


class AugmentGenerator(tf.keras.utils.Sequence):
    """Data generator with on-the-fly rotation (0/90/180/270) and transpose."""
    def __init__(self, images, labels, batch_size, augment=True, samples_per_epoch_per_class=None):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.samples_per_epoch_per_class = samples_per_epoch_per_class

        # Group indices by class for per-class subset sampling
        self.class_indices = {}
        for idx, label in enumerate(labels):
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
        images = self.images[batch_idx].copy()
        labels = self.labels[batch_idx]

        if self.augment:
            for i in range(len(images)):
                k = np.random.randint(0, 4)  # 0-3 rotations
                if k > 0:
                    images[i] = np.rot90(images[i], k=k)
                if np.random.random() < 0.5:
                    images[i] = np.transpose(images[i], (1, 0, 2))

        return images, labels

    def on_epoch_end(self):
        self._sample_indices()


# ─── Load dataset ──────────────────────────────────────────────────
from utils.mnist_loader import load_data

(x_train, y_train), (x_test, y_test) = load_data(DATA_DIR, NUM_CLASSES, flip_pixels=7*28, max_samples_per_class=MAX_SAMPLES_PER_CLASS, pbm_dir=PBM_DIR)

print(f"Train: {len(x_train)}, Test: {len(x_test)}")

# ─── Build & train model ───────────────────────────────────────────
L = tf.keras.layers
# Residual Conv2D backbone + multi-scale feedback
inputs = L.Input(shape=(28, 28, 1))
x = L.GaussianNoise(GAUSSIAN_NOISE)(inputs)

# CBAM: Convolutional Block Attention Module
def channel_attention(x, reduction=8):
    """Channel Attention (SE) - avg+max pooling."""
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
    """Spatial Attention - scale-aware kernel size."""
    h = int(x.shape[1])
    k = 5 if h >= 28 else 3

    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    attn = L.Conv2D(1, k, padding='same', activation='sigmoid')(concat)
    return tf.multiply(x, attn)

def cbam_block(x):
    """CBAM: Channel attention -> Spatial attention."""
    x = channel_attention(x)
    x = spatial_attention(x)
    x = L.LayerNormalization()(x)
    return x

# Conv block 0: 28x28 -> 28x28
c0_res = x
c0 = L.SeparableConv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY), name='percept_c0')(x)
c0 = L.SeparableConv2D(64, 3, padding='same', activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(c0)
c0 = L.LayerNormalization()(c0)
c0 = L.Add()([c0, c0_res])
c0 = L.Activation('relu')(c0)
c0 = L.SpatialDropout2D(SPATIAL_DROPOUT)(c0)
c0 = L.Add()([c0, cbam_block(c0)])

# Conv block 1: 28x28 -> 14x14 with residual skip
c1_res = L.SeparableConv2D(96, 3, strides=2, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY), name='percept_c1')(c0)
c1 = L.SeparableConv2D(96, 3, padding='same', activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(c1_res)
c1 = L.LayerNormalization()(c1)
skip_28_to_14 = L.Conv2D(96, 1, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(c0)
c1 = L.Add()([c1, skip_28_to_14])
c1 = L.Activation('relu')(c1)
c1 = L.SpatialDropout2D(SPATIAL_DROPOUT)(c1)
c1 = L.Add()([c1, cbam_block(c1)])

# Conv block 2: 14x14 → 7x7 with residual skip
c2_res = L.SeparableConv2D(128, 3, strides=2, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY), name='percept_c2')(c1)
c2 = L.SeparableConv2D(128, 3, padding='same', activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(c2_res)
c2 = L.LayerNormalization()(c2)
skip_14_to_7 = L.Conv2D(128, 1, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(c1)
c2 = L.Add()([c2, skip_14_to_7])
c2 = L.Activation('relu')(c2)
c2 = L.SpatialDropout2D(SPATIAL_DROPOUT)(c2)
c2 = L.Add()([c2, cbam_block(c2)])

# Conv block 3: 7x7 → 7x7 (no pooling)
c3_res = L.SeparableConv2D(128, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY), name='percept_c3')(c2)
c3 = L.SeparableConv2D(128, 3, padding='same', activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(c3_res)
c3 = L.LayerNormalization()(c3)
c3 = L.Add()([c3, c3_res])
c3 = L.Activation('relu')(c3)
c3 = L.SpatialDropout2D(SPATIAL_DROPOUT)(c3)
c3 = L.LayerNormalization()(c3)

# Cross-scale residual connections: 7x7 -> 14x14 -> 28x28
up14 = L.UpSampling2D(2)(c3)
up14 = L.Conv2D(96, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(up14)
up14 = L.LayerNormalization()(up14)
up14 = L.Activation('relu')(up14)
c1_res = L.Add()([c1, up14])
c1_res = L.Dropout(DROPOUT)(c1_res)
c1_res = L.LayerNormalization()(c1_res)

up28 = L.UpSampling2D(2)(c1_res)
up28 = L.Conv2D(64, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(up28)
up28 = L.LayerNormalization()(up28)
up28 = L.Activation('relu')(up28)
c0_res = L.Add()([c0, up28])
c0_res = L.Dropout(DROPOUT)(c0_res)
c0_res = L.LayerNormalization()(c0_res)

# Multi-scale features for classification
pool7 = L.GlobalAveragePooling2D()(c3)
pool14 = L.GlobalAveragePooling2D()(c1_res)
pool28 = L.GlobalAveragePooling2D()(c0_res)

# Branch A: fine-scale path from Flatten(c3)
branch_a = L.Flatten()(c3)
branch_a = L.Dense(DENSE_SIZE, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(branch_a)
branch_a = L.Dropout(DROPOUT)(branch_a)
p_a = L.Dense(DENSE_SIZE, kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(branch_a)
branch_a = L.Add()([branch_a, p_a])
branch_a = L.Activation('relu')(branch_a)
branch_a = L.Dense(DENSE_SIZE, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(branch_a)
branch_a = L.Dropout(DROPOUT)(branch_a)
p_a = L.Dense(DENSE_SIZE, kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(branch_a)
branch_a = L.Add()([branch_a, p_a])
branch_a = L.Activation('relu')(branch_a)
logits_a = L.Dense(NUM_CLASSES)(branch_a)

# Branch B: cross-scale path from pool7 + pool14 + pool28
branch_b = L.Concatenate()([pool7, pool14, pool28])
branch_b = L.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(branch_b)
branch_b = L.Dropout(DROPOUT)(branch_b)
branch_b = L.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(branch_b)
branch_b = L.Dropout(DROPOUT)(branch_b)
logits_b = L.Dense(NUM_CLASSES)(branch_b)

# Final logits with learned weighting via additive merge
outputs = L.Add()([logits_a, logits_b])
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# ─── GCE Loss ────────────────────────────────────────────────────────
def gce_loss(q=0.5):
    """Generalized Cross Entropy loss."""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, NUM_CLASSES)
        p = tf.nn.softmax(y_pred)
        p_t = tf.reduce_sum(y_true_one_hot * p, axis=-1)
        p_t_safe = tf.clip_by_value(p_t, 1e-7, 1.0)
        # GCE: (1 - p_t^q) / q
        return tf.reduce_mean((1.0 - tf.pow(p_t_safe, q)) / q)
    return loss_fn

loss_fn = gce_loss(q=GCE_Q)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LR, clipnorm=1.0),
              loss=loss_fn, metrics=['accuracy'])

class _GradNormObserver:
    def __init__(self, optimizer, output_var):
        self.optimizer = optimizer
        self._output_var = output_var
        self._original_apply = optimizer.apply_gradients

        def _wrapped_apply(grads_and_vars, **kwargs):
            if self.optimizer is None:
                return None
            grads = [g for g, _ in grads_and_vars if g is not None]
            if grads:
                gn = tf.linalg.global_norm(grads)
            else:
                gn = tf.constant(0.0, dtype=tf.float32)
            self._output_var.assign(gn)
            return self._original_apply(grads_and_vars, **kwargs)

        optimizer.apply_gradients = _wrapped_apply


class GradNormEMACallback(tf.keras.callbacks.Callback):
    def __init__(self, decay=0.95, target=1.0, factor=0.01, min_lr=3e-6, max_lr=2e-3, start_epoch=0, batch_interval=1):
        super().__init__()
        self.decay = decay
        self.target = target
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.start_epoch = start_epoch
        self.batch_interval = batch_interval
        self.ema_norm = 0.0
        self._observer = None
        self._epoch = None
        self._norm_var = None
        self._optimizer = None
        self._batch = 0

    def set_model(self, model):
        super().set_model(model)
        optimizer = model.optimizer
        self._norm_var = tf.Variable(0.0, trainable=False)
        self._observer = _GradNormObserver(optimizer, self._norm_var)
        self._optimizer = optimizer

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch
        self._batch = 0

    def on_train_batch_end(self, batch, logs=None):
        if self._epoch is None or self._epoch < self.start_epoch:
            return
        if self._norm_var is None:
            return
        norm = float(self._norm_var.numpy())
        self.ema_norm = self.decay * self.ema_norm + (1.0 - self.decay) * norm
        self._batch += 1
        if self.batch_interval <= 0 or (self._batch % self.batch_interval) == 0:
            current_lr = float(tf.keras.backend.get_value(self._optimizer.learning_rate))
            ema = float(self.ema_norm)
            delta = (self.target - ema) / (ema + 1e-8)
            new_lr = current_lr * (1.0 + self.factor * delta)
            new_lr = max(self.min_lr, min(self.max_lr, new_lr))
            self._optimizer.learning_rate.assign(new_lr)
            #print(f"  GradNormEMA batch: ema_norm={ema:.4f}, lr={new_lr:.6f}")

    def on_epoch_end(self, epoch, logs=None):
        return


total_steps = len(x_train) // BATCH_SIZE

class GaussianNoiseDecay(tf.keras.callbacks.Callback):
    def __init__(self, initial_noise, decay_epochs):
        super().__init__()
        self.initial_noise = initial_noise
        self.decay_epochs = decay_epochs
        self.noise_layer = None
    def on_train_begin(self, logs=None):
        for layer in self.model.layers:
            if isinstance(layer, L.GaussianNoise):
                self.noise_layer = layer
                break
    def on_epoch_begin(self, epoch, logs=None):
        if self.noise_layer is None:
            return
        progress = min(1.0, epoch / self.decay_epochs)
        noise = self.initial_noise + (GAUSSIAN_NOISE_END - self.initial_noise) * progress
        noise = max(0.0, noise)
        self.noise_layer.stddev = noise
        print(f"  Gaussian noise: {noise:.6f}")

class DropoutDecay(tf.keras.callbacks.Callback):
    def __init__(self, start, end, decay_epochs):
        super().__init__()
        self.start = start
        self.end = end
        self.decay_epochs = decay_epochs
        self._layers = []
    def on_train_begin(self, logs=None):
        self._layers = [layer for layer in self.model.layers if isinstance(layer, L.Dropout)]
    def on_epoch_begin(self, epoch, logs=None):
        progress = min(1.0, epoch / self.decay_epochs)
        rate = self.start + (self.end - self.start) * progress
        for layer in self._layers:
            layer.rate = float(rate)
        print(f"  Dropout rate: {rate:.4f}")

class BestEpochLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        self.best_epoch = 0
        self.best_loss = float('inf')
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
    def on_train_end(self, logs=None):
        print(f"\n═══ Best epoch: {self.best_epoch} (val_loss: {self.best_loss:.4f}) ═══")


callbacks = [
    GradNormEMACallback(decay=GRAD_NORM_EMA_DECAY, target=GRAD_NORM_TARGET, factor=GRAD_ADAPTIVE_FACTOR, min_lr=GRAD_ADAPTIVE_MIN_LR, max_lr=GRAD_ADAPTIVE_MAX_LR, start_epoch=0, batch_interval=GRAD_ADAPTIVE_BATCH_INTERVAL),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True),
    GaussianNoiseDecay(GAUSSIAN_NOISE, GAUSSIAN_NOISE_DECAY_EPOCHS),
    BestEpochLogger(),
    tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
]

if AUGMENT:
    train_gen = AugmentGenerator(x_train, y_train, BATCH_SIZE, augment=True, samples_per_epoch_per_class=SAMPLES_PER_EPOCH_PER_CLASS)
    model.fit(train_gen, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=callbacks)
else:
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), callbacks=callbacks)

model.save('mnist_model')
print("\\nModel saved to mnist_model/")
