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
NUM_CLASSES = 10
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.003
LR_WARMUP_EPOCHS = 7
LR_MIN = 0.00008
LR_DECAY_EPOCHS = 26
BATCH_DECAY_FACTOR = 0.05
DROPOUT = 0.20
FOCAL_GAMMA = 2
FOCAL_ALPHA = 0.24
LABEL_SMOOTHING = 0.01
EARLY_STOP_PATIENCE = 8
GAUSSIAN_NOISE = 0.006
GAUSSIAN_NOISE_DECAY_EPOCHS = 50
GAUSSIAN_NOISE_END = 0.0001
LOG_DIR = "logs/run4"


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
# Residual Conv2D with label smoothing and LR warmup
inputs = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.GaussianNoise(GAUSSIAN_NOISE)(inputs)

# Conv block 1: 28x28 → 14x14
c1 = tf.keras.layers.SeparableConv2D(64, 3, padding='same', activation='relu')(x)
c1 = tf.keras.layers.MaxPooling2D(2)(c1)
c1 = tf.keras.layers.Dropout(DROPOUT)(c1)
# SE block 1 (channel attention)
se1 = tf.keras.layers.GlobalAveragePooling2D()(c1)
se1 = tf.keras.layers.Dense(64 // 4, activation='relu')(se1)
se1 = tf.keras.layers.Dense(64, activation='sigmoid')(se1)
se1 = tf.keras.layers.Reshape((1, 1, 64))(se1)
c1 = tf.keras.layers.Multiply()([c1, se1])
# Spatial attention 1
sa1 = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c1)
c1 = tf.keras.layers.Multiply()([c1, sa1])

# Conv block 2: 14x14 → 7x7
c2 = tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation='relu')(c1)
c2 = tf.keras.layers.MaxPooling2D(2)(c2)
c2 = tf.keras.layers.Dropout(DROPOUT)(c2)
# SE block 2 (channel attention)
se2 = tf.keras.layers.GlobalAveragePooling2D()(c2)
se2 = tf.keras.layers.Dense(128 // 4, activation='relu')(se2)
se2 = tf.keras.layers.Dense(128, activation='sigmoid')(se2)
se2 = tf.keras.layers.Reshape((1, 1, 128))(se2)
c2 = tf.keras.layers.Multiply()([c2, se2])
# Spatial attention 2
sa2 = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(c2)
c2 = tf.keras.layers.Multiply()([c2, sa2])

# Conv block 3: 7x7 → 7x7 (no pooling)
c3 = tf.keras.layers.SeparableConv2D(128, 3, padding='same', activation='relu')(c2)
c3 = tf.keras.layers.Dropout(DROPOUT)(c3)

# Flatten + Dense with 2 residual connections
x = tf.keras.layers.Flatten()(c3)
x = tf.keras.layers.GaussianNoise(GAUSSIAN_NOISE)(x)
d1 = tf.keras.layers.Dense(512, activation='relu')(x)
d1 = tf.keras.layers.Dropout(DROPOUT)(d1)
p1 = tf.keras.layers.Dense(512)(x)
x = tf.keras.layers.Add()([d1, p1])
x = tf.keras.layers.Activation('relu')(x)
# 2nd residual connection
d2 = tf.keras.layers.Dense(512, activation='relu')(x)
d2 = tf.keras.layers.Dropout(DROPOUT)(d2)
p2 = tf.keras.layers.Dense(512)(x)
x = tf.keras.layers.Add()([d2, p2])
x = tf.keras.layers.Activation('relu')(x)

outputs = tf.keras.layers.Dense(NUM_CLASSES)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# ─── Focal Loss with Label Smoothing ────────────────────────────────
def focal_loss_with_smoothing(gamma=2.0, alpha=0.25, smoothing=0.1):
    """Focal loss with label smoothing."""
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, NUM_CLASSES)
        # Label smoothing: 0.9 for correct class, 0.1/9 for others
        y_true_smooth = y_true_one_hot * (1.0 - smoothing) + smoothing / NUM_CLASSES
        ce = tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred, from_logits=True)
        p = tf.exp(-ce)
        focal_weight = alpha * tf.pow(1.0 - p, gamma)
        return tf.reduce_mean(focal_weight * ce)
    return loss_fn

loss_fn = focal_loss_with_smoothing(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA, smoothing=LABEL_SMOOTHING)
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
            if isinstance(layer, tf.keras.layers.GaussianNoise):
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

model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test), callbacks=callbacks)

# ─── Stats ─────────────────────────────────────────────────────────
print("\n─── Per-class stats ───")
preds = model.predict(x_test).argmax(axis=1)
for c in range(NUM_CLASSES):
    mask = y_test == c
    class_total = mask.sum()
    class_correct = (preds[mask] == c).sum()
    class_acc = class_correct / class_total if class_total > 0 else 0
    probs = tf.nn.softmax(model.predict(x_test[mask])).numpy()
    avg_conf = probs[:, c].mean() if class_total > 0 else 0
    print(f"  Class {class_correct}/{class_total} correct, avg conf: {avg_conf:.3f}")

# ─── Confusion Matrix ──────────────────────────────────────────────
print("\n─── Confusion Matrix (predicted →) ───")
print("     " + "  ".join(f"{c:3d}" for c in range(NUM_CLASSES)))
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
for true_c in range(NUM_CLASSES):
    mask = y_test == true_c
    for pred_c in range(NUM_CLASSES):
        cm[true_c, pred_c] = int((preds[mask] == pred_c).sum())
for true_c in range(NUM_CLASSES):
    row = "  ".join(f"{cm[true_c, pred_c]:3d}" for pred_c in range(NUM_CLASSES))
    print(f"  {true_c}: {row}")

model.save('mnist_model')
print("\nModel saved to mnist_model/")
