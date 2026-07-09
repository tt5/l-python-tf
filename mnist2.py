#!/usr/bin/env python3
"""mnist2.py

Train a classifier on 22x22 binary PBM images from data/pbm/.
Upscales to 28x28 using nearest-neighbor (same as pbm_utils.upscale),
binarizes, and trains an 11-class classifier.

Labels 0-9: predicted digit from filename
Label 10: low confidence (confidence < 0.6)

Usage:
    /home/n/miniconda3/envs/tf/bin/python mnist2.py
"""

import argparse
import tensorflow as tf
import numpy as np
import sys
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
            return super()._resource_apply_dense(grad, var, apply_state)

        def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
            if self.weight_decay > 0:
                grad = grad + tf.gather(self.weight_decay * var, indices)
            return super()._resource_apply_sparse(grad, var, indices, apply_state)

DEFAULT_PBM_DIR = Path(__file__).resolve().parent / "../../telephonegame/data/pbm"
TB_LOG_DIR = Path(__file__).resolve().parent / "logs" / "mnist2"

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=DEFAULT_PBM_DIR, help="PBM data directory")
args, _ = parser.parse_known_args()
PBM_DIR = args.data_dir
IMG_SIZE = 22
TARGET_SIZE = 28
LOW_CONF_THRESHOLD = 0.6
EPOCHS = 160
DROPOUT = 0.35 # unused
SPATIAL_DROPOUT = 0.38
DENSE_DROPOUT = 0.08
L2_REG = 0.0001
BATCH_SIZE = 256
BASE_LR = 0.0007
WARMUP_EPOCHS = 4
DECAY_EPOCHS = EPOCHS - WARMUP_EPOCHS
LR_START = 0.00002
LR_END = 0.00009
POLY_EPSILON = 2
AUGMENT = True  # On-the-fly augmentation: random rotate/transpose


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


def symmetric_cross_entropy(alpha=0.5, beta=0.5):
    """Symmetric Cross Entropy loss."""
    def loss_fn(y_true, y_pred):
        # Standard CE: true labels vs predictions
        ce_true = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        
        # Reverse CE: predictions vs true labels
        probs = tf.nn.softmax(y_pred, axis=-1)
        y_true_one_hot = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), depth=tf.shape(y_pred)[-1])
        ce_reverse = tf.keras.losses.categorical_crossentropy(y_true_one_hot, probs)
        
        return alpha * tf.reduce_mean(ce_true) + beta * tf.reduce_mean(ce_reverse)
    return loss_fn



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
        # Binarize (should already be binary from upscale, but ensure)
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

    print(f"\\nDerived NUM_CLASSES = {NUM_CLASSES} (max_label={max_label}, low_conf={low_conf_label})")

    print(f"\nLabel distribution (low_conf threshold={LOW_CONF_THRESHOLD}):")
    for c in range(NUM_CLASSES):
        count = np.sum(labels == c)
        if c < NUM_CLASSES:
            name = f"{c}"
        else:
            name = "low_conf"
        print(f"  {name}: {count}")

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

    inputs = L.Input(shape=(TARGET_SIZE, TARGET_SIZE))
    x = L.Reshape((TARGET_SIZE, TARGET_SIZE, 1))(inputs)

    # Conv block 0: 28x28 -> 28x28
    c0 = L.SeparableConv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    c0 = L.SpatialDropout2D(SPATIAL_DROPOUT)(c0)
    c0 = L.Dropout(DENSE_DROPOUT)(c0)

    # Conv block 1: 28x28 -> 14x14
    c1 = L.SeparableConv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(c0)
    c1 = L.MaxPooling2D(2)(c1)
    c1 = L.SpatialDropout2D(SPATIAL_DROPOUT)(c1)
    c1 = L.Dropout(DENSE_DROPOUT)(c1)
    c1 = cbam_block(c1)

    # Conv block 2: 14x14 -> 7x7
    c2 = L.SeparableConv2D(128, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='conv14')(c1)
    c2 = L.MaxPooling2D(2)(c2)
    c2 = L.SpatialDropout2D(SPATIAL_DROPOUT)(c2)
    c2 = L.Dropout(DENSE_DROPOUT)(c2)
    c2 = cbam_block(c2)

    # Conv block 3: 7x7 → 7x7 (no pooling)
    c3 = L.SeparableConv2D(128, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name='conv7_deep2')(c2)
    c3 = L.SpatialDropout2D(SPATIAL_DROPOUT)(c3)
    c3 = L.Dropout(DENSE_DROPOUT)(c3)
    c3 = cbam_block(c3)

    # Multi-scale kernel block at 7x7
    b1 = L.Conv2D(16, 1, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(c3)
    b3 = L.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(c3)
    b5 = L.Conv2D(16, 5, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(c3)
    c3 = L.Concatenate()([b1, b3, b5])
    c3 = L.Conv2D(128, 1, padding='same', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(c3)
    c3 = L.LayerNormalization()(c3)
    c3 = L.Dropout(DENSE_DROPOUT/2)(c3)

    # Flatten + Dense with 2 residual connections
    x = L.Flatten()(c3)

    d1 = L.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    d1 = L.Dropout(DENSE_DROPOUT)(d1)
    p1 = L.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = L.Add()([d1, p1])
    x = L.Activation('relu')(x)
    # 2nd residual connection
    d2 = L.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    d2 = L.Dropout(DENSE_DROPOUT)(d2)
    p2 = L.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = L.Add()([d2, p2])
    x = L.Activation('relu')(x)
    # 3rd residual connection
    d3 = L.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    d3 = L.Dropout(DENSE_DROPOUT)(d3)
    p3 = L.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    x = L.Add()([d3, p3])
    x = L.Activation('relu')(x)

    outputs = L.Dense(NUM_CLASSES)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=AdamW(clipnorm=1.0, weight_decay=1e-5, learning_rate=BASE_LR), loss=symmetric_cross_entropy(), metrics=['accuracy'])

    print("\nModel summary:")
    model.summary()

    print(f"\nTraining for {EPOCHS} epochs...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=150,
        restore_best_weights=True
    )
    lr_schedule = LinearLRSchedule(
        lr_start=LR_START,
        lr_peak=BASE_LR,
        lr_end=LR_END,
        warmup_epochs=WARMUP_EPOCHS,
        decay_epochs=DECAY_EPOCHS
    )
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=str(TB_LOG_DIR), update_freq='epoch')

    if AUGMENT:
        train_gen = AugmentGenerator(x_train, y_train, batch_size=BATCH_SIZE, augment=True)
        history = model.fit(train_gen, epochs=EPOCHS, validation_data=(x_test, y_test),
                  callbacks=[early_stopping, lr_schedule, tb_callback])
    else:
        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(x_test, y_test),
                  callbacks=[early_stopping, lr_schedule, tb_callback])

    stopped_epoch = getattr(early_stopping, 'stopped_epoch', None)
    best_epoch = getattr(early_stopping, 'best_epoch', None)
    print("\nEarlyStopping summary:")
    if best_epoch is not None:
        print(f"  Best epoch: {best_epoch + 1}")
    if stopped_epoch is not None:
        print(f"  Stopped epoch: {stopped_epoch + 1}")

    print("\nFinal evaluation:")
    model.evaluate(x_test, y_test, verbose=2)

    # Save
    save_path = Path(__file__).parent / "mnist2_model"
    model.save(str(save_path))
    print(f"\nModel saved to {save_path}")

if __name__ == "__main__":
    main()
