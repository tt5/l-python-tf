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
NUM_CLASSES = 5
EPOCHS = 60
BATCH_SIZE = 256
BASE_LR = 0.0005
GRAD_ADAPTIVE_MIN_LR = 0.00001
GRAD_ADAPTIVE_MAX_LR = 0.0005
GRAD_ADAPTIVE_FACTOR = 0.00009
GRAD_ADAPTIVE_BATCH_INTERVAL = 3
GRAD_NORM_EMA_DECAY = 0.94
GRAD_NORM_TARGET = 1.0
DROPOUT = 0.45
DENSE_SIZE = 512
L2_DECAY = 0.00005
GCE_Q = 0.5
EARLY_STOP_PATIENCE = 100
GAUSSIAN_NOISE = 0.005
GAUSSIAN_NOISE_DECAY_EPOCHS = 50
GAUSSIAN_NOISE_END = 0.00005
SPATIAL_DROPOUT = 0.3
LOG_DIR = "logs/cls1b"


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
    """Load synthetic classes from disk, then synthesize black/white classes."""
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    max_train, max_test = 0, 0
    loaded_classes = 0

    for class_id in range(NUM_CLASSES - 2):
        class_dir = DATA_DIR / f"class_{class_id}"
        if not class_dir.exists():
            print(f"ERROR: Directory not found: {class_dir}")
            sys.exit(1)
        files = sorted(class_dir.glob("*.npy"))
        if len(files) == 0:
            print(f"ERROR: No .npy files in {class_dir}")
            sys.exit(1)

        split_idx = int(len(files) * 0.8)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        for f in train_files:
            img = np.load(str(f)).astype(np.float32) / 255.0
            img = np.asarray(img).reshape(28, 28, 1)
            train_images.append(img)
            train_labels.append(class_id)

        for f in test_files:
            img = np.load(str(f)).astype(np.float32) / 255.0
            img = np.asarray(img).reshape(28, 28, 1)
            test_images.append(img)
            test_labels.append(class_id)

        if len(train_files) > max_train:
            max_train = len(train_files)
        if len(test_files) > max_test:
            max_test = len(test_files)

        loaded_classes += 1
        print(f"Class {class_id}: {len(train_files)} train, {len(test_files)} test")

    # Synthesize black/white classes to match the largest existing per-class counts
    black = np.zeros((28, 28, 1), dtype=np.float32)
    white = np.ones((28, 28, 1), dtype=np.float32)
    for img, label in [(black, NUM_CLASSES - 2), (white, NUM_CLASSES - 1)]:
        for i in range(max_train):
            train_images.append(img)
            train_labels.append(label)
        for i in range(max_test):
            test_images.append(img)
            test_labels.append(label)
        print(f"Class {label}: {max_train} train, {max_test} test")

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

# Conv block 1: 28x28 → 14x14 with residual skip
c1 = L.SeparableConv2D(64, 3, strides=2, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
skip_28_to_14 = L.Conv2D(64, 1, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(x)
c1 = L.Add()([c1, skip_28_to_14])
c1 = L.Dropout(DROPOUT)(c1)
c1 = cbam_block(c1)

# Conv block 2: 14x14 → 7x7 with residual skip
c2 = L.SeparableConv2D(128, 3, strides=2, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(c1)
skip_14_to_7 = L.Conv2D(128, 1, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_DECAY))(c1)
c2 = L.Add()([c2, skip_14_to_7])
c2 = L.Dropout(DROPOUT)(c2)
c2 = cbam_block(c2)
c2 = L.SpatialDropout2D(SPATIAL_DROPOUT)(c2)

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
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=BASE_LR, clipvalue=1.0),
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
    train_gen = AugmentGenerator(x_train, y_train, BATCH_SIZE, augment=True)
    model.fit(train_gen, epochs=EPOCHS, validation_data=(x_test, y_test), callbacks=callbacks)
else:
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), callbacks=callbacks)

model.save('mnist_model')
print("\\nModel saved to mnist_model/")
