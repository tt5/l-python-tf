#!/usr/bin/env python3
"""train_cvae.py

Simple Conditional VAE for synthetic pictogram data.
Compatible with convert_generator.py for ONNX conversion.

Usage:
    python train_cvae.py
"""

import numpy as np
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─── Config ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic" / "data"
LATENT_DIM = 512
EPOCHS = 20
BATCH_SIZE = 256
NUM_CLASSES = 10
LR = 0.002
LR_WARMUP_START = 0.0002
LR_WARMUP_EPOCHS = 4
LR_DECAY_EPOCHS = EPOCHS - LR_WARMUP_EPOCHS
LR_END = 0.00005
KL_WARMUP_START = 0.00005
KL_WEIGHT_START = 0.42
KL_WEIGHT_TARGET = 0.5
KL_WARMUP_EPOCHS = 8
KL_DECAY_EPOCHS = EPOCHS - KL_WARMUP_EPOCHS
PIXEL_LOSS_WEIGHT = 0.65
PERCEPTUAL_LOSS_WEIGHT = 1.35
GRAD_NOISE_SCALE = 0.015
GRAD_NOISE_DECAY_EPOCHS = EPOCHS
INFO_NCE_WEIGHT = 0.11
TEMPERATURE = 0.5
DROPOUT = 0.06


# ─── Data ──────────────────────────────────────────────────────────
def load_data():
    x_train, y_train_oh, x_test, y_test_oh = [], [], [], []
    for c in range(NUM_CLASSES):
        files = sorted((DATA_DIR / f"class_{c}").glob("*.npy"))
        split = int(len(files) * 0.5)
        for f in files[:split]:
            img = np.load(str(f)).astype(np.float32) / 255.0
            img = (img > 0.5).astype(np.float32)  # binary thresholding
            x_train.append(img)
            y_train_oh.append(c)
        for f in files[split:]:
            img = np.load(str(f)).astype(np.float32) / 255.0
            img = (img > 0.5).astype(np.float32)  # binary thresholding
            x_test.append(img)
            y_test_oh.append(c)

    x_train = np.array(x_train).reshape(-1, 28, 28, 1)
    x_test = np.array(x_test).reshape(-1, 28, 28, 1)
    y_train = keras.utils.to_categorical(y_train_oh, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test_oh, NUM_CLASSES)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data()
print(f"Train: {len(x_train)}, Test: {len(x_test)}")


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
    c1 = layers.Conv2D(64, 3, strides=2, padding="same")(img)
    c1 = layers.Activation("relu")(c1)
    # SE attention 1
    se1 = layers.GlobalAveragePooling2D()(c1)
    se1 = layers.Dense(64 // 4, activation="relu")(se1)
    se1 = layers.Dense(64, activation="sigmoid")(se1)
    se1 = layers.Reshape((1, 1, 64))(se1)
    c1 = layers.Multiply()([c1, se1])

    # Conv block 2: 14x14 → 7x7
    c2 = layers.Conv2D(128, 3, strides=2, padding="same")(c1)
    c2 = layers.Activation("relu")(c2)
    # Residual connection: project c1 (14x14x64) to match c2 (7x7x128)
    skip = layers.Conv2D(128, 1, strides=2, padding="same")(c1)
    c2 = layers.Add()([c2, skip])
    c2 = layers.Activation("relu")(c2)
    # SE attention 2
    se2 = layers.GlobalAveragePooling2D()(c2)
    se2 = layers.Dense(128 // 4, activation="relu")(se2)
    se2 = layers.Dense(128, activation="sigmoid")(se2)
    se2 = layers.Reshape((1, 1, 128))(se2)
    c2 = layers.Multiply()([c2, se2])

    # Conv block 3: 7x7 → 7x7 (stride 1)
    c3 = layers.Conv2D(256, 3, strides=1, padding="same")(c2)
    c3 = layers.Activation("relu")(c3)
    # Residual connection: project c2 (7x7x128) to match c3 (7x7x256)
    skip2 = layers.Conv2D(256, 1, padding="same")(c2)
    c3 = layers.Add()([c3, skip2])
    c3 = layers.Activation("relu")(c3)
    # SE attention 3
    se3 = layers.GlobalAveragePooling2D()(c3)
    se3 = layers.Dense(256 // 4, activation="relu")(se3)
    se3 = layers.Dense(256, activation="sigmoid")(se3)
    se3 = layers.Reshape((1, 1, 256))(se3)
    c3 = layers.Multiply()([c3, se3])

    # Bottleneck
    x = layers.Flatten()(c3)
    x = layers.Concatenate()([x, lbl])
    x = layers.Dense(768, activation="relu")(x)
    # Self-attention bottleneck (treat as sequence of tokens)
    x = layers.Reshape((1, 768))(x)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=192)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(DROPOUT)(x)
    z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return keras.Model([img, lbl], [z_mean, z_log_var, z], name="encoder")


# ─── Upsample via Sub-pixel Convolution ────────────────────────────
def upsample(x, target_size):
    """Upsample using sub-pixel convolution (depth_to_space)."""
    channels = x.shape[-1]
    x = layers.Conv2D(channels * 4, 3, padding='same', activation='relu')(x)
    x = tf.nn.depth_to_space(x, 2)
    return x

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
    x = layers.Dense(7 * 7 * 256, activation="relu")(z)
    x = layers.Reshape((7, 7, 256))(x)
    # FiLM at 7x7
    gamma7, beta7 = make_film(256)(lbl)
    gamma7 = layers.Reshape((1, 1, 256))(gamma7)
    beta7 = layers.Reshape((1, 1, 256))(beta7)
    x = layers.Multiply()([x, gamma7])
    x = layers.Add()([x, beta7])
    # SE attention on initial projection
    se_init = layers.GlobalAveragePooling2D()(x)
    se_init = layers.Dense(256 // 4, activation="relu")(se_init)
    se_init = layers.Dense(256, activation="sigmoid")(se_init)
    se_init = layers.Reshape((1, 1, 256))(se_init)
    x = layers.Multiply()([x, se_init])
    # Spatial attention before conv
    sa_init = layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, sa_init])

    # 7x7 → 14x14
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
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
    skip_7 = layers.Conv2D(128, 1, padding="same")(x)
    x = upsample(x, [14, 14])
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
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

    # 14x14 → 28x28
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
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
    skip_14 = layers.Conv2D(64, 1, padding="same")(x)
    x = upsample(x, [28, 28])
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    # FiLM at output resolution
    gamma28c, beta28c = make_film(64)(lbl)
    gamma28c = layers.Reshape((1, 1, 64))(gamma28c)
    beta28c = layers.Reshape((1, 1, 64))(beta28c)
    x = layers.Multiply()([x, gamma28c])
    x = layers.Add()([x, beta28c])
    # Spatial attention
    sa28_pre = layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, sa28_pre])

    x = layers.Conv2D(1, 3, padding="same", activation="sigmoid", dtype="float32")(x)
    return keras.Model([z, lbl], x, name="generator")


# ─── Perceptual Loss Feature Extractor (Multi-layer) ───────────────
def build_feature_extractor():
    """Use the trained MNIST classifier as multi-layer feature extractor."""
    mnist_path = Path(__file__).resolve().parent / "mnist_model"
    if mnist_path.exists():
        cls_model = keras.models.load_model(str(mnist_path), custom_objects={'loss_fn': lambda y_true, y_pred: keras.losses.categorical_crossentropy(y_true, y_pred)})
        # Extract features from multiple layers (after each conv block)
        # Use GlobalAvgPool on each spatial layer to get fixed-size vectors
        feature_layers = []
        for layer in cls_model.layers:
            if isinstance(layer, (layers.Conv2D, layers.SeparableConv2D)):
                feature_layers.append(layer.output)
        if feature_layers:
            # Global average pool each feature map
            pooled = []
            for f in feature_layers:
                if len(f.shape) == 4:  # (batch, h, w, c)
                    pooled.append(layers.GlobalAveragePooling2D()(f))
                else:
                    pooled.append(f)
            feat_model = keras.Model(cls_model.input, pooled, name="feature_extractor")
            return feat_model
    # Fallback: simple CNN
    inp = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inp)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    return keras.Model(inp, [x], name="feature_extractor")


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
        warmup_epochs = tf.cast(KL_WARMUP_EPOCHS, tf.float32)
        decay_epochs = tf.cast(KL_DECAY_EPOCHS, tf.float32)
        # Phase 1: Warmup from KL_WARMUP_START → KL_WEIGHT_START
        warmup_progress = tf.minimum(1.0, epoch / warmup_epochs)
        warmup_weight = KL_WARMUP_START + (KL_WEIGHT_START - KL_WARMUP_START) * warmup_progress
        # Phase 2: Linear decay KL_WEIGHT_START → KL_WEIGHT_TARGET
        decay_epoch = epoch - warmup_epochs
        decay_progress = tf.minimum(1.0, decay_epoch / decay_epochs)
        decay_weight = KL_WEIGHT_START + (KL_WEIGHT_TARGET - KL_WEIGHT_START) * decay_progress
        # Select phase
        return tf.where(epoch < warmup_epochs, warmup_weight, decay_weight)

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
            pixel_loss = tf.reduce_mean(tf.reduce_sum(
                keras.losses.binary_crossentropy(images, reconstruction), axis=(1, 2)))

            # Perceptual loss (multi-layer feature-level)
            real_features = self.feature_extractor(images)
            fake_features = self.feature_extractor(reconstruction)
            perceptual_loss = 0.0
            for rf, ff in zip(real_features, fake_features):
                perceptual_loss += tf.reduce_mean(tf.square(rf - ff))
            perceptual_loss /= len(real_features)

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

            # Combined loss
            total_loss = (PIXEL_LOSS_WEIGHT * pixel_loss +
                          PERCEPTUAL_LOSS_WEIGHT * perceptual_loss +
                          kl_weight * kl_loss +
                          INFO_NCE_WEIGHT * info_nce_loss)

        grads = tape.gradient(total_loss, self.trainable_weights)
        grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in grads]
        # Linear decay of gradient noise
        noise_scale = GRAD_NOISE_SCALE * (1.0 - tf.cast(self.epoch_tracker, tf.float32) / tf.cast(GRAD_NOISE_DECAY_EPOCHS, tf.float32))
        noise_scale = tf.maximum(0.0, noise_scale)
        grads = [g + tf.random.normal(tf.shape(g), stddev=noise_scale) if g is not None else g for g in grads]
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
        pixel_loss = tf.reduce_mean(tf.reduce_sum(
            keras.losses.binary_crossentropy(images, reconstruction), axis=(1, 2)))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        total_loss = pixel_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(pixel_loss)
        self.kl_loss_tracker.update_state(kl_loss)
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
        val_loss = logs.get('val_total_loss')
        if val_loss is not None and val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = epoch
    def on_train_end(self, logs=None):
        print(f"\n═══ Best epoch: {self.best_epoch} (val_loss: {self.best_loss:.4f}) ═══")


best_generator_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_generator.h5")

class GeneratorCheckpoint(keras.callbacks.Callback):
    def __init__(self, generator, filepath, monitor='val_total_loss'):
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
    keras.callbacks.EarlyStopping(monitor='val_total_loss', patience=10, restore_best_weights=True),
    GeneratorCheckpoint(generator, best_generator_path),
    BestEpochLogger(vae),
    keras.callbacks.TensorBoard(log_dir='logs/run2', histogram_freq=1),
]

history = vae.fit(x_train, y_train, validation_data=(x_test, y_test),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=2)

# ─── Load best generator weights (in case EarlyStopping didn't trigger) ──
if os.path.exists(best_generator_path):
    generator.load_weights(best_generator_path)
    print(f"\nLoaded best generator weights from {best_generator_path}")

# ─── Save ─────────────────────────────────────────────────────────
save_dir = os.path.dirname(os.path.abspath(__file__))
generator.save(os.path.join(save_dir, "cvae_generator.h5"))
print(f"\nGenerator saved to {save_dir}/cvae_generator.h5")
