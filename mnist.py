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
from pathlib import Path

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic" / "data"
NUM_CLASSES = 6
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.005
LR_WARMUP_EPOCHS = 7
LR_MIN = 0.0001
LR_DECAY_EPOCHS = 26
BATCH_DECAY_FACTOR = 0.05
DROPOUT = 0.40
DENSE_SIZE = 512
L2_DECAY = 1e-5
GCE_Q = 0.5
EARLY_STOP_PATIENCE = 8
GAUSSIAN_NOISE = 0.005
GAUSSIAN_NOISE_DECAY_EPOCHS = 50
GAUSSIAN_NOISE_END = 0.0001
LOG_DIR = "logs/run5"


AUGMENT = True  # On-the-fly augmentation: random rotate/transpose


class AugmentGenerator(tf.keras.utils.Sequence):
    """Data generator with on-the-fly rotation (0/90/180/270) and transpose."""
    def __init__(self, images, labels, batch_size, augment=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(images))
    
    def __len__(self):
        return len(self.images) // self.batch_size
    
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
        np.random.shuffle(self.indices)


def load_synthetic():
    """Load all synthetic classes from .npy files."""
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for class_id in range(NUM_CLASSES):
        class_dir = DATA_DIR / f"class_{class_id}"
        if not class_dir.exists():
            print(f"ERROR: Directory not found: {class_dir}")
            sys.exit(1)

        files = sorted(class_dir.glob("*.npy"))
        if len(files) == 0:
            print(f"ERROR: No .npy files in {class_dir}")
            sys.exit(1)

        # Split 50% train, 50% test
        split_idx = int(len(files) * 0.5)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        for f in train_files:
            img = np.load(str(f)).astype(np.float32) / 255.0
            train_images.append(img)
            train_labels.append(class_id)

        for f in test_files:
            img = np.load(str(f)).astype(np.float32) / 255.0
            test_images.append(img)
            test_labels.append(class_id)

        print(f"Class {class_id}: {len(train_files)} train, {len(test_files)} test")

    train_x = np.array(train_images).reshape(-1, 28, 28, 1)
    train_y = np.array(train_labels, dtype=np.int64)
    test_x = np.array(test_images).reshape(-1, 28, 28, 1)
    test_y = np.array(test_labels, dtype=np.int64)

    # Shuffle
    train_perm = np.random.permutation(len(train_x))
    test_perm = np.random.permutation(len(test_x))
    train_x, train_y = train_x[train_perm], train_y[train_perm]
    test_x, test_y = test_x[test_perm], test_y[test_perm]

    return (train_x, train_y), (test_x, test_y)


# ─── Load dataset ──────────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = load_synthetic()

print(f"Train: {len(x_train)}, Test: {len(x_test)}")

# ─── Build & train model ───────────────────────────────────────────
# Residual Conv2D with label smoothing and LR warmu
L = tf.keras.layers
inputs = L.Input(shape=(28, 28, 1))
x = L.GaussianNoise(GAUSSIAN_NOISE)(inputs)

# CBAM: Convolutional Block Attention Module
def channel_attention(x, reduction=16):
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
    """Spatial Attention - Conv2D 7x7."""
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    attn = L.Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
    return tf.multiply(x, attn)

def cbam_block(x):
    """CBAM: Channel attention -> Spatial attention."""
    x = channel_attention(x)
    x = spatial_attention(x)
    return x

# Conv block 1: 28x28 → 14x14
c1 = L.SeparableConv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
c1 = L.MaxPooling2D(2)(c1)
c1 = L.Dropout(DROPOUT)(c1)
c1 = cbam_block(c1)

# Conv block 2: 14x14 → 7x7
c2 = L.SeparableConv2D(128, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(c1)
c2 = L.MaxPooling2D(2)(c2)
c2 = L.Dropout(DROPOUT)(c2)
c2 = cbam_block(c2)

# Conv block 3: 7x7 → 7x7 (no pooling)
c3 = L.SeparableConv2D(128, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(c2)
c3 = L.Dropout(DROPOUT)(c3)
c3 = cbam_block(c3)

# Flatten + Dense with 2 residual connections
x = L.Flatten()(c3)
x = L.GaussianNoise(GAUSSIAN_NOISE)(x)
d1 = L.Dense(DENSE_SIZE, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
d1 = L.Dropout(DROPOUT)(d1)
p1 = L.Dense(DENSE_SIZE, kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
x = L.Add()([d1, p1])
x = L.Activation('relu')(x)
# 2nd residual connection
d2 = L.Dense(DENSE_SIZE, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
d2 = L.Dropout(DROPOUT)(d2)
p2 = L.Dense(DENSE_SIZE, kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
x = L.Add()([d2, p2])
x = L.Activation('relu')(x)

outputs = L.Dense(NUM_CLASSES)(x)
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
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipvalue=1.0),
              loss=loss_fn, metrics=['accuracy'])

# ─── LR Warmup + Decay Callback ────────────────────────────────────
class LRWarmupDecay(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=5, target_lr=1e-3, decay_epochs=50, end_lr=1e-5, steps_per_epoch=100, batch_decay_factor=0.5):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.decay_epochs = decay_epochs
        self.end_lr = end_lr
        self.steps_per_epoch = steps_per_epoch
        self.current_step = 0
        self.batch_decay_factor = batch_decay_factor

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            self.epoch_start_lr = self.target_lr * (epoch + 1) / self.warmup_epochs
        else:
            decay_epoch = epoch - self.warmup_epochs
            decay_progress = min(1.0, decay_epoch / self.decay_epochs)
            self.epoch_start_lr = self.target_lr + (self.end_lr - self.target_lr) * decay_progress
        self.epoch_end_lr = self.epoch_start_lr
        if self.current_epoch + 1 < self.warmup_epochs:
            self.epoch_end_lr = self.target_lr * (self.current_epoch + 2) / self.warmup_epochs
        elif self.current_epoch + 1 >= self.warmup_epochs:
            next_epoch = self.current_epoch + 1
            decay_epoch = next_epoch - self.warmup_epochs
            decay_progress = min(1.0, decay_epoch / self.decay_epochs)
            self.epoch_end_lr = self.target_lr + (self.end_lr - self.target_lr) * decay_progress
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.epoch_start_lr)
        print(f"  LR: {self.epoch_start_lr:.6f} → {self.epoch_end_lr:.6f}")

    def on_train_batch_begin(self, batch, logs=None):
        self.current_step += 1
        batch_progress = batch / max(1, self.steps_per_epoch - 1)
        # Per-batch decay goes below epoch_end_lr by batch_decay_factor
        lr_min = self.epoch_end_lr - (self.epoch_start_lr - self.epoch_end_lr) * self.batch_decay_factor
        lr = self.epoch_start_lr + (lr_min - self.epoch_start_lr) * batch_progress
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)

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
    LRWarmupDecay(warmup_epochs=LR_WARMUP_EPOCHS, target_lr=LEARNING_RATE, decay_epochs=LR_DECAY_EPOCHS, end_lr=LR_MIN, steps_per_epoch=total_steps, batch_decay_factor=BATCH_DECAY_FACTOR),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True),
    GaussianNoiseDecay(GAUSSIAN_NOISE, GAUSSIAN_NOISE_DECAY_EPOCHS),
    BestEpochLogger(),
    tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
]

if AUGMENT:
    train_gen = AugmentGenerator(x_train, y_train, BATCH_SIZE, augment=True)
    model.fit(train_gen, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=callbacks)
else:
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), callbacks=callbacks)

model.save('mnist_model')
print("\\nModel saved to mnist_model/")
