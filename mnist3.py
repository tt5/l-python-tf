#!/usr/bin/env python3
"""mnist3.py

Train a classifier on 22x22 binary PBM images from data/pbm2/.
Upscales to 28x28 using nearest-neighbor (same as pbm_utils.upscale),
binarizes, and trains a 12-class classifier.

Labels 0-9: predicted digit from filename
Label 10: low confidence (confidence < 0.6)
Label 11: low confidence 2 (confidence < 0.6 variant)

Usage:
    /home/n/miniconda3/envs/tf/bin/python mnist3.py
"""

import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path

try:
    from tensorflow_addons.optimizers import AdamW
except Exception:
    class AdamW(tf.keras.optimizers.Adam):
        def __init__(self, weight_decay=0.0, **kwargs):
            super().__init__(**kwargs)
            self.weight_decay = weight_decay

        def _resource_apply_dense(self, grad, var, apply_state=None):
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * var
            return super()._resource_apply_dense(grad, var, apply_state=apply_state)

        def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
            if self.weight_decay > 0:
                grad = grad + tf.gather(self.weight_decay * var, indices)
            return super()._resource_apply_sparse(grad, var, indices, apply_state=apply_state)

DEFAULT_PBM_DIR = Path(__file__).resolve().parent / "../../telephonegame/data/pbm2"
TB_LOG_DIR = Path(__file__).resolve().parent / "logs" / "mnist3"

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=DEFAULT_PBM_DIR, help="PBM data directory")
args, _ = parser.parse_known_args()
PBM_DIR = args.data_dir
IMG_SIZE = 22
TARGET_SIZE = 28
LOW_CONF_THRESHOLD = 0.6
EPOCHS = 50
DROPOUT = 0.3
BASE_LR = 1e-3
POLY_EPSILON = 2
BATCH_SIZE = 128
AUGMENT = True  # On-the-fly augmentation: random rotate/transpose
GRAD_NORM_EMA_DECAY = 0.97
GRAD_NORM_TARGET = 1.0
GRAD_ADAPTIVE_FACTOR = 0.0019
GRAD_ADAPTIVE_BATCH_INTERVAL = 2
GRAD_ADAPTIVE_MIN_LR = 0.000005
GRAD_ADAPTIVE_MAX_LR = 0.02


def poly_loss(labels, logits, epsilon=None):
    """CE + 0.25 * (1 - p_t)^2."""
    if epsilon is None:
        epsilon = POLY_EPSILON
    ce = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    probs = tf.nn.softmax(logits, axis=-1)
    labels_one_hot = tf.one_hot(tf.cast(labels, tf.int32), depth=tf.shape(logits)[-1])
    p_t = tf.reduce_sum(probs * labels_one_hot, axis=-1)
    p_t = tf.clip_by_value(p_t, 1e-7, 1.0)
    pt_loss = 0.25 * tf.pow(1.0 - p_t, 2)
    return tf.reduce_mean(ce + epsilon * pt_loss)


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
            print(f"  GradNormEMA batch: ema_norm={ema:.4f}, lr={new_lr:.6f}")

    def on_epoch_end(self, epoch, logs=None):
        return


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
    """Nearest-neighbor block upscale (same as pbm_utils.upscale)."""
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


class AugmentGenerator(tf.keras.utils.Sequence):
    """On-the-fly augmentation: random rotate 90° and transpose."""
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
                k = np.random.randint(0, 4)
                if k > 0:
                    images[i] = np.rot90(images[i], k=k)
                if np.random.random() < 0.5:
                    images[i] = np.transpose(images[i])

        return images, labels

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def main():
    files = sorted(PBM_DIR.glob("*.pbm"))
    print(f"Loading {len(files)} PBM images from {PBM_DIR}...")

    images = []
    labels = []
    low_conf_count = 0

    for i, f in enumerate(files):
        parts = f.stem.split('_')
        predicted_digit = int(parts[0])
        confidence = float(parts[1]) / 10000.0

        img = load_pbm(f)
        img_28 = upscale(img, TARGET_SIZE, TARGET_SIZE)
        img_28 = (img_28 > 0.5).astype(np.float32)

        images.append(img_28)

        if confidence < LOW_CONF_THRESHOLD:
            labels.append(-1)  # placeholder, will be remapped
            low_conf_count += 1
        else:
            labels.append(predicted_digit)

        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1}/{len(files)}")

    images = np.array(images)
    labels = np.array(labels)

    # Derive num_classes from data: max label + 1 for low_conf
    max_label = labels[labels >= 0].max()
    low_conf_label = max_label + 1
    labels[labels == -1] = low_conf_label
    NUM_CLASSES = low_conf_label + 1

    print(f"\nDerived NUM_CLASSES = {NUM_CLASSES} (max_label={max_label}, low_conf={low_conf_label})")

    print(f"\nLabel distribution (low_conf threshold={LOW_CONF_THRESHOLD}):")
    for c in range(NUM_CLASSES):
        count = np.sum(labels == c)
        if c < NUM_CLASSES:
            name = f"{c}"
        else:
            name = "low_conf"
        print(f"  {name}: {count}")
    print(f"  low_conf_count={low_conf_count}")

    # Shuffle
    np.random.seed(42)
    perm = np.random.permutation(len(images))
    images = images[perm]
    labels = labels[perm]

    # Train/test split (80/20)
    split = int(len(images) * 0.8)
    x_train, x_test = images[:split], images[split:]
    y_train, y_test = labels[:split], labels[split:]
    print(f"\nTrain: {len(x_train)}, Test: {len(x_test)}")

    # Build model
    L = tf.keras.layers
    inputs = L.Input(shape=(TARGET_SIZE, TARGET_SIZE))
    x = L.Reshape((TARGET_SIZE, TARGET_SIZE, 1))(inputs)
    x = L.Conv2D(32, 7, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = L.Conv2D(64, 3, strides=2, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5), name='conv14')(x)
    x = L.LayerNormalization()(x)
    x = L.Conv2D(64, 3, strides=2, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5), name='conv7_base')(x)
    x = L.LayerNormalization()(x)
    x = L.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5), name='conv7_deep1')(x)
    x = L.LayerNormalization()(x)
    x = L.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-5), name='conv7_deep2')(x)
    x = L.LayerNormalization()(x)
    # Pooled conv output at 14x14 and 7x7 for multi-scale feature extraction
    x = L.Flatten()(x)
    x = L.Dense(384, activation='relu')(x)
    x = L.Dropout(DROPOUT)(x)
    x_main = L.Dense(192, activation='relu')(x)
    x = L.Dense(192)(x)
    x = L.Add()([x_main, x])
    x = L.Activation('relu')(x)

    # Squeeze-and-excitation on 192-d features
    se = L.Dense(192 // 16, activation='relu')(x)
    se = L.Dense(192, activation='sigmoid')(se)
    x = L.Multiply()([x, se])

    x = L.Dropout(DROPOUT)(x)
    outputs = L.Dense(NUM_CLASSES)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=AdamW(clipnorm=1.0, weight_decay=1e-5, learning_rate=BASE_LR), loss=poly_loss, metrics=['accuracy'])

    print("\nModel summary:")
    model.summary()

    print(f"\nTraining for {EPOCHS} epochs...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    grad_norm_cb = GradNormEMACallback(
        decay=GRAD_NORM_EMA_DECAY,
        target=GRAD_NORM_TARGET,
        factor=GRAD_ADAPTIVE_FACTOR,
        min_lr=GRAD_ADAPTIVE_MIN_LR,
        max_lr=GRAD_ADAPTIVE_MAX_LR,
        start_epoch=0,
        batch_interval=GRAD_ADAPTIVE_BATCH_INTERVAL,
    )
    if AUGMENT:
        train_gen = AugmentGenerator(x_train, y_train, batch_size=BATCH_SIZE, augment=True)
        model.fit(train_gen, epochs=EPOCHS, validation_data=(x_test, y_test),
                  callbacks=[early_stopping, grad_norm_cb, tf.keras.callbacks.TensorBoard(log_dir=str(TB_LOG_DIR), update_freq='epoch')],
                  verbose=2)
    else:
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(x_test, y_test),
                  callbacks=[early_stopping, grad_norm_cb, tf.keras.callbacks.TensorBoard(log_dir=str(TB_LOG_DIR), update_freq='epoch')])

    print("\nFinal evaluation:")
    model.evaluate(x_test, y_test, verbose=2)

    # Save
    save_path = Path(__file__).parent / "mnist3_model"
    model.save(str(save_path))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
