#!/usr/bin/env python3
"""cls2enc.py

Train a classifier with the same conditional trunk as train_cvae2.py's encoder,
so weights can be transferred to the CVAE encoder later.

Usage:
    /home/n/miniconda3/envs/tf/bin/python cls2enc.py
"""

import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path

DEFAULT_PBM_DIR = Path(__file__).resolve().parent / "../../telephonegame/data/pbm"
TB_LOG_DIR = Path(__file__).resolve().parent / "logs" / "cls2enc"

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=Path, default=DEFAULT_PBM_DIR, help="PBM data directory")
args, _ = parser.parse_known_args()
PBM_DIR = args.data_dir

# ─── Hyperparameters ──────────────────────────────────────────────
LATENT_DIM = 256
EPOCHS = 150
BATCH_SIZE = 128
NUM_CLASSES = None  # derived
TARGET_SIZE = 28
LOW_CONF_THRESHOLD = 0.6
L2_REG = 0.00001
SPATIAL_DROPOUT = 0.3
DENSE_DROPOUT = 0.25
AUGMENT = True
LEARNING_RATE = 0.0003
LR_WARMUP_EPOCHS = 11
LR_START = 0.00005
LR_DECAY_EPOCHS = EPOCHS - LR_WARMUP_EPOCHS
LR_END = 0.0000015


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
            grid[y, x] = 1.0 - bit
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
        img_28 = (img_28 > 0.5).astype(np.float32)

        images.append(img_28)

        if confidence < LOW_CONF_THRESHOLD:
            labels.append(-1)
        else:
            labels.append(predicted_digit)

        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1}/{len(files)}")

    images = np.array(images)[..., np.newaxis]
    labels = np.array(labels)

    global NUM_CLASSES
    max_label = labels[labels >= 0].max()
    low_conf_label = max_label + 1
    labels[labels == -1] = low_conf_label
    NUM_CLASSES = low_conf_label + 1

    print(f"\nDerived NUM_CLASSES = {NUM_CLASSES} (max_label={max_label}, low_conf={low_conf_label})")

    labels_oh = tf.keras.utils.to_categorical(labels, NUM_CLASSES)

    np.random.seed(42)
    perm = np.random.permutation(len(images))
    images = images[perm]
    labels_oh = labels_oh[perm]

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


# ─── Conditional blocks matching train_cvae2.py encoder ───────────
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
    attn = tf.keras.layers.Dropout(DENSE_DROPOUT)(attn)

    ctx = attn * tf.expand_dims(v, 1)
    ctx = tf.reshape(ctx, [B, H, W, channels])
    ctx = tf.keras.layers.LayerNormalization(center=False, scale=False)(ctx)

    return tf.keras.layers.Add()([spatial, ctx])


def make_film(channels):
    def film_fn(label):
        emb = tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(label)
        emb = tf.keras.layers.Dropout(DENSE_DROPOUT)(emb)
        gamma = tf.keras.layers.Dense(channels, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(emb)
        beta = tf.keras.layers.Dense(channels, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(emb)
        gamma = tf.keras.layers.Reshape((1, 1, channels))(gamma)
        beta = tf.keras.layers.Reshape((1, 1, channels))(beta)
        return gamma, beta
    return film_fn


def build_classifier():
    image_input = tf.keras.layers.Input(shape=(28, 28, 1), name="image_input")
    label_input = tf.keras.layers.Input(shape=(NUM_CLASSES,), name="label_input")

    # 28x28 block
    x = tf.keras.layers.Conv2D(32, 3, strides=1, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv1")(image_input)
    x = tf.keras.layers.LayerNormalization(name="enc_ln1")(x)
    x28_pre = x

    x28 = label_cross_attention(x28_pre, label_input, 32, L2_REG)
    film_gamma28, film_beta28 = make_film(32)(label_input)
    x28 = x28 * (1.0 + film_gamma28) + film_beta28
    x28 = tf.keras.layers.SpatialDropout2D(SPATIAL_DROPOUT, name="enc_drop28")(x28)

    x28 = tf.keras.layers.Add()([
        x28,
        tf.keras.layers.Conv2D(32, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_skip28_1x1")(x28_pre),
    ])

    x28_res = x28
    x28 = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_res28_1")(x28)
    x28 = tf.keras.layers.LayerNormalization(name="enc_ln28_2")(x28)
    x28 = tf.keras.layers.Conv2D(32, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_res28_2")(x28)
    x28 = tf.keras.layers.LayerNormalization(name="enc_ln28_3")(x28)
    x28 = tf.keras.layers.Add()([x28, x28_res])
    x28 = tf.keras.layers.Activation("relu", name="enc_act28")(x28)

    x = tf.keras.layers.MaxPooling2D(2, name="enc_pool28")(x28)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv14")(x)
    x = tf.keras.layers.LayerNormalization(name="enc_ln14_1")(x)
    x = tf.keras.layers.Add()([
        x,
        tf.keras.layers.Conv2D(64, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_skip14_1x1")(
            tf.keras.layers.MaxPooling2D(2)(x28)
        ),
    ])
    x14 = x

    x14_pre = x14
    x14 = label_cross_attention(x14, label_input, 64, L2_REG)
    film_gamma14, film_beta14 = make_film(64)(label_input)
    x14 = x14 * (1.0 + film_gamma14) + film_beta14
    x14 = tf.keras.layers.Add()([
        x14,
        tf.keras.layers.Conv2D(64, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_skip14_2_1x1")(x14_pre),
    ])

    x14_res = x14
    x14 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_res14_1")(x14)
    x14 = tf.keras.layers.LayerNormalization(name="enc_ln14_2")(x14)
    x14 = tf.keras.layers.Conv2D(64, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_res14_2")(x14)
    x14 = tf.keras.layers.LayerNormalization(name="enc_ln14_3")(x14)
    x14 = tf.keras.layers.Add()([x14, x14_res])
    x14 = tf.keras.layers.Activation("relu", name="enc_act14")(x14)

    x = tf.keras.layers.Conv2D(64, 3, strides=1, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv14_2")(x14)
    x = tf.keras.layers.LayerNormalization(name="enc_ln14_4")(x)
    x = tf.keras.layers.SpatialDropout2D(SPATIAL_DROPOUT, name="enc_drop14_2")(x)
    x = tf.keras.layers.Conv2D(96, 3, strides=1, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_conv96")(x)
    x = tf.keras.layers.LayerNormalization(name="enc_ln96_1")(x)
    x = tf.keras.layers.MaxPooling2D(2, name="enc_pool96")(x)
    x = tf.keras.layers.Add()([
        x,
        tf.keras.layers.Conv2D(96, 1, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_skip96_1x1")(
            tf.keras.layers.MaxPooling2D(2)(tf.keras.layers.Activation("relu")(x14))
        ),
    ])

    x_res = x
    x = tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_res96_1")(x)
    x = tf.keras.layers.LayerNormalization(name="enc_ln96_2")(x)
    x = tf.keras.layers.Conv2D(96, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_res96_2")(x)
    x = tf.keras.layers.LayerNormalization(name="enc_ln96_3")(x)
    x = tf.keras.layers.Add()([x, x_res])
    x = tf.keras.layers.Activation("relu", name="enc_act96")(x)
    x = tf.keras.layers.SpatialDropout2D(SPATIAL_DROPOUT, name="enc_drop96")(x)

    # SE block
    se = tf.keras.layers.GlobalAveragePooling2D(name="enc_gap")(x)
    se = tf.keras.layers.Dense(96 // 4, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_se1")(se)
    se = tf.keras.layers.Dense(96, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_se2")(se)
    x = tf.keras.layers.Multiply(name="enc_se_mult")([x, se])
    x = tf.keras.layers.SpatialDropout2D(SPATIAL_DROPOUT, name="enc_drop_se")(x)
    x = tf.keras.layers.Flatten(name="enc_flat")(x)

    label_proj = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_label_proj")(label_input)
    label_proj = tf.keras.layers.Dropout(DENSE_DROPOUT, name="enc_label_drop")(label_proj)

    x = tf.keras.layers.Concatenate(name="enc_concat")([x, label_proj])
    x = tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_dense512")(x)
    x = tf.keras.layers.LayerNormalization(name="enc_ln512_1")(x)
    x = tf.keras.layers.Dropout(DENSE_DROPOUT, name="enc_drop512")(x)
    x = tf.keras.layers.LayerNormalization(name="enc_ln512_2")(x)
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2_REG), name="enc_dense256")(x)
    x = tf.keras.layers.LayerNormalization(name="enc_ln256")(x)
    x = tf.keras.layers.Dropout(DENSE_DROPOUT, name="enc_drop256")(x)

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="enc_logits")(x)
    model = tf.keras.Model(inputs=[image_input, label_input], outputs=outputs, name="cls2enc")
    return model


class LinearWarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, lr_start, lr_peak, lr_end, warmup_epochs=0, total_epochs=0, log_dir=None):
        super().__init__()
        self.lr_start = float(lr_start)
        self.lr_peak = float(lr_peak)
        self.lr_end = float(lr_end)
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.log_dir = str(log_dir) if log_dir else None
        self._writer = None

    def set_model(self, model):
        super().set_model(model)
        if self.log_dir:
            self._writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            progress = epoch / max(1, self.warmup_epochs - 1) if self.warmup_epochs > 1 else 1.0
            lr = self.lr_start + (self.lr_peak - self.lr_start) * progress
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs - 1)
            progress = max(0.0, min(1.0, progress))
            lr = self.lr_end + 0.5 * (self.lr_peak - self.lr_end) * (1.0 + tf.math.cos(progress * 3.141592653589793))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, float(lr))

    def on_epoch_end(self, epoch, logs=None):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"  epoch={epoch + 1:03d} lr={lr:.8f}")
        if self._writer is not None:
            with self._writer.as_default():
                tf.summary.scalar('lr', lr, step=epoch)


def main():
    x_train, y_train, x_test, y_test = load_dataset()

    model = build_classifier()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss="categorical_crossentropy",
        metrics=['accuracy']
    )

    print("\nModel summary:")
    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=150,
        restore_best_weights=True
    )
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=str(TB_LOG_DIR), update_freq='epoch')
    lr_cb = LinearWarmupCosineDecay(
        lr_start=LR_START,
        lr_peak=LEARNING_RATE,
        lr_end=LR_END,
        warmup_epochs=LR_WARMUP_EPOCHS,
        total_epochs=EPOCHS,
        log_dir=TB_LOG_DIR,
    )

    print(f"\nTraining cls2enc for {EPOCHS} epochs...")
    model.fit(
        [x_train, y_train], y_train,
        validation_data=([x_test, y_test], y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, lr_cb, tb_callback],
        verbose=2
    )

    print("\nFinal evaluation:")
    model.evaluate([x_test, y_test], y_test, verbose=2)

    save_path = Path(__file__).parent / "cls2enc_model"
    model.save(str(save_path))
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    main()
