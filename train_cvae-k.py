#!/usr/bin/env python3
"""train_cvae.py

Train a Conditional Variational Autoencoder (CVAE) with adversarial loss
on synthetic pictogram data.
Loads data from synthetic dataset (classes 0-10).

Usage:
    python train_cvae.py

Saves:
    - cvae_encoder.h5
    - cvae_generator.h5
"""

import tensorflow as tf
import numpy as np
import os
import sys
from pathlib import Path

# Mixed precision training for faster training on GPU
tf.keras.mixed_precision.set_global_policy('mixed_float16')

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic" / "data"
LATENT_DIM = 256
EPOCHS = 50
BATCH_SIZE = 128
NUM_CLASSES = 10
LABEL_SMOOTHING = 0.05
KL_WEIGHT_START = 0.001
KL_WEIGHT_END = 0.9
KL_ANNEAL_EPOCHS = 18
LR_MAX = 0.000003
LR_MIN = 0.000003
LR_CYCLE_LENGTH = 2
LR_WARMUP_EPOCHS = 15 
ADV_WEIGHT = 8  # Weight for adversarial loss
FM_WEIGHT = 8  # Weight for feature matching loss
EMA_DECAY = 0.999  # Exponential moving average decay for generator weights
G_STEPS_PER_D_STEP = 4  # Number of generator updates per discriminator update

class KLAnnealing(tf.keras.callbacks.Callback):
    def __init__(self, weight_start, weight_end, anneal_epochs):
        super().__init__()
        self.weight_start = weight_start
        self.weight_end = weight_end
        self.anneal_epochs = anneal_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.anneal_epochs:
            progress = epoch / self.anneal_epochs
            new_weight = self.weight_start + (self.weight_end - self.weight_start) * progress
        else:
            new_weight = self.weight_end
        self.model.kl_weight.assign(new_weight)
        print(f"  KL weight: {new_weight:.4f}")


def load_synthetic():
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

        split_idx = int(len(files) * 0.8)
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

    train_perm = np.random.permutation(len(train_x))
    test_perm = np.random.permutation(len(test_x))
    train_x, train_y = train_x[train_perm], train_y[train_perm]
    test_x, test_y = test_x[test_perm], test_y[test_perm]

    return (train_x, train_y), (test_x, test_y)


(x_train, y_train), (x_test, y_test) = load_synthetic()
print(f"Train: {len(x_train)}, Test: {len(x_test)}")


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim), dtype=z_mean.dtype)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder():
    image_input = tf.keras.layers.Input(shape=(28, 28, 1), name="image_input")
    label_input = tf.keras.layers.Input(shape=(NUM_CLASSES,), name="label_input")
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(image_input)
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Concatenate()([x, label_input])
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    z_mean = tf.keras.layers.Dense(LATENT_DIM, name="z_mean")(x)
    z_log_var = tf.keras.layers.Dense(LATENT_DIM, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model([image_input, label_input], [z_mean, z_log_var, z], name="encoder")


def build_generator():
    latent_input = tf.keras.layers.Input(shape=(LATENT_DIM,), name="latent_input")
    label_input = tf.keras.layers.Input(shape=(NUM_CLASSES,), name="label_input")

    label_emb_7x7 = tf.keras.layers.Dense(7 * 7 * 16, activation="relu")(label_input)
    label_emb_7x7 = tf.keras.layers.Reshape((7, 7, 16))(label_emb_7x7)

    label_emb_14x14 = tf.keras.layers.Dense(14 * 14 * 16, activation="relu")(label_input)
    label_emb_14x14 = tf.keras.layers.Reshape((14, 14, 16))(label_emb_14x14)

    label_emb_28x28 = tf.keras.layers.Dense(28 * 28 * 16, activation="relu")(label_input)
    label_emb_28x28 = tf.keras.layers.Reshape((28, 28, 16))(label_emb_28x28)

    x = tf.keras.layers.Concatenate()([latent_input, label_input])
    x = tf.keras.layers.Dense(7 * 7 * 64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Concatenate()([x, label_emb_7x7])

    x = tf.keras.layers.Conv2D(64, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Concatenate()([x, label_emb_14x14])
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Concatenate()([x, label_emb_28x28])
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    x = tf.keras.layers.Activation("linear", dtype="float32")(x)
    return tf.keras.Model([latent_input, label_input], x, name="generator")


def build_discriminator():
    image_input = tf.keras.layers.Input(shape=(28, 28, 1), name="image_input")
    label_input = tf.keras.layers.Input(shape=(NUM_CLASSES,), name="label_input")

    # Label embedding spatially
    label_emb = tf.keras.layers.Dense(28 * 28, activation="relu")(label_input)
    label_emb = tf.keras.layers.Reshape((28, 28, 1))(label_emb)

    # Concatenate image and label
    x = tf.keras.layers.Concatenate()([image_input, label_emb])  # (28, 28, 2)

    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same")(x)   # 14x14
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same")(x)  # 7x7
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding="same")(x)  # 4x4
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    features = tf.keras.layers.Conv2D(512, 3, strides=2, padding="same")(x)  # 2x2
    features = tf.keras.layers.LeakyReLU(0.2)(features)

    # Output both features and prediction
    flat = tf.keras.layers.Flatten()(features)
    pred = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(flat)
    return tf.keras.Model([image_input, label_input], [pred, features], name="discriminator")


class CVAE(tf.keras.Model):
    def __init__(self, encoder, generator, discriminator, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.adv_loss_tracker = tf.keras.metrics.Mean(name="adv_loss")
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.kl_weight = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        # EMA for generator weights
        self.ema_decay = EMA_DECAY
        self.ema_weights = [tf.Variable(w, trainable=False) for w in self.generator.trainable_weights]

    def update_ema(self):
        """Update EMA weights after each training step."""
        for ema_w, w in zip(self.ema_weights, self.generator.trainable_weights):
            ema_w.assign(self.ema_decay * ema_w + (1.0 - self.ema_decay) * w)

    def apply_ema_weights(self):
        """Apply EMA weights to the generator (for inference/saving)."""
        for ema_w, w in zip(self.ema_weights, self.generator.trainable_weights):
            w.assign(ema_w)

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker,
                self.adv_loss_tracker, self.d_loss_tracker]

    def compile(self, optimizer=None, d_optimizer=None, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        self.g_optimizer = optimizer
        self.d_optimizer = d_optimizer

    def train_step(self, data):
        images, labels = data
        labels_oh = tf.one_hot(labels, NUM_CLASSES)
        labels_oh = labels_oh * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / NUM_CLASSES

        # ─── Train discriminator ────────────────────────────────────
        with tf.GradientTape() as d_tape:
            z_mean, z_log_var, z = self.encoder([images, labels_oh])
            z = z + tf.random.normal(tf.shape(z), dtype=z.dtype) * 0.1
            fake_images = self.generator([z, labels_oh])

            real_pred, real_features = self.discriminator([images, labels_oh])
            fake_pred, fake_features = self.discriminator([fake_images, labels_oh])

            d_loss_real = tf.keras.losses.binary_crossentropy(tf.ones_like(real_pred), real_pred)
            d_loss_fake = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_pred), fake_pred)
            d_loss = tf.cast(tf.reduce_mean(d_loss_real + d_loss_fake), tf.float32)
            scaled_d_loss = self.d_optimizer.get_scaled_loss(d_loss)

        scaled_d_grads = d_tape.gradient(scaled_d_loss, self.discriminator.trainable_weights)
        d_grads = self.d_optimizer.get_unscaled_gradients(scaled_d_grads)
        d_grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in d_grads]
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        # ─── Train generator (multiple steps per discriminator update) ──
        for _ in range(G_STEPS_PER_D_STEP):
            with tf.GradientTape() as g_tape:
                z_mean, z_log_var, z = self.encoder([images, labels_oh])
                z = z + tf.random.normal(tf.shape(z), dtype=z.dtype) * 0.1
                reconstruction = self.generator([z, labels_oh])

                recon_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.keras.losses.binary_crossentropy(images, reconstruction),
                        axis=(1, 2)
                    )
                )
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(
                        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                        axis=1
                    )
                )

                # Adversarial loss: fool discriminator
                fake_pred, fake_features = self.discriminator([reconstruction, labels_oh])
                _, real_features = self.discriminator([images, labels_oh])

                adv_loss = tf.cast(
                    tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_pred), fake_pred)),
                    tf.float32
                )

                # Feature matching loss: match intermediate feature statistics
                fm_loss = tf.cast(tf.reduce_mean(tf.square(
                    tf.reduce_mean(real_features, axis=0) - tf.reduce_mean(fake_features, axis=0)
                )), tf.float32)

                total_loss = (tf.cast(recon_loss, tf.float32) +
                             tf.cast(self.kl_weight, tf.float32) * tf.cast(kl_loss, tf.float32) +
                             ADV_WEIGHT * adv_loss +
                             FM_WEIGHT * fm_loss)
                scaled_g_loss = self.g_optimizer.get_scaled_loss(total_loss)

            scaled_g_grads = g_tape.gradient(scaled_g_loss, self.encoder.trainable_weights + self.generator.trainable_weights)
            g_grads = self.g_optimizer.get_unscaled_gradients(scaled_g_grads)
            g_grads = [tf.clip_by_value(g, -1.0, 1.0) if g is not None else g for g in g_grads]
            self.g_optimizer.apply_gradients(
                zip(g_grads, self.encoder.trainable_weights + self.generator.trainable_weights)
            )

            # Update EMA weights
            self.update_ema()

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.adv_loss_tracker.update_state(adv_loss)
        self.d_loss_tracker.update_state(d_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        labels_oh = tf.one_hot(labels, NUM_CLASSES)
        labels_oh = labels_oh * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / NUM_CLASSES
        z_mean, z_log_var, z = self.encoder([images, labels_oh])
        reconstruction = self.generator([z, labels_oh])
        recon_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(images, reconstruction),
                axis=(1, 2)
            )
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )
        total_loss = tf.cast(recon_loss, tf.float32) + tf.cast(kl_loss, tf.float32)
        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

    def generate(self, label, noise=None):
        label_oh = tf.keras.utils.to_categorical([label], NUM_CLASSES)
        if noise is None:
            noise = tf.random.normal(shape=(1, LATENT_DIM))
        return self.generator([noise, label_oh])


# ─── Build & train ────────────────────────────────────────────────
encoder = build_encoder()
generator = build_generator()
discriminator = build_discriminator()
cvae = CVAE(encoder, generator, discriminator)
cvae.compile(
    optimizer=tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(LR_MAX)),
    d_optimizer=tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(LR_MAX * 0.5)),
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_total_loss', patience=100, restore_best_weights=True),
    KLAnnealing(KL_WEIGHT_START, KL_WEIGHT_END, KL_ANNEAL_EPOCHS),
]

class LRScheduler(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs=1, total_epochs=60, max_lr=1e-3, min_lr=1e-4, cycle_length=10):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_length = cycle_length

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Linear warmup from 1% of max_lr to max_lr
            warmup_start = self.max_lr * 0.01
            new_lr = warmup_start + (self.max_lr - warmup_start) * (epoch + 1) / self.warmup_epochs
        else:
            cycle_position = (epoch - self.warmup_epochs) % self.cycle_length
            half_cycle = self.cycle_length / 2
            if cycle_position < half_cycle:
                progress = cycle_position / half_cycle
            else:
                progress = 1.0 - (cycle_position - half_cycle) / half_cycle
            new_lr = self.min_lr + (self.max_lr - self.min_lr) * progress
        tf.keras.backend.set_value(cvae.g_optimizer.inner_optimizer.learning_rate, new_lr)
        tf.keras.backend.set_value(cvae.d_optimizer.inner_optimizer.learning_rate, new_lr * 0.5)
        print(f"  LR: {new_lr:.6f}")

callbacks.insert(0, LRScheduler(
    warmup_epochs=LR_WARMUP_EPOCHS,
    total_epochs=EPOCHS,
    max_lr=LR_MAX,
    min_lr=LR_MIN,
    cycle_length=LR_CYCLE_LENGTH,
))

print(f"\nTraining CVAE-GAN for {EPOCHS} epochs...")
history = cvae.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
)

print("Encoder summary:")
encoder.summary()
print("\nGenerator summary:")
generator.summary()
print("\nDiscriminator summary:")
discriminator.summary()

print("\n─── Training stats ───")
print(f"  Final train loss: {history.history['total_loss'][-1]:.4f}")
print(f"  Final val loss: {history.history['val_total_loss'][-1]:.4f}")
print(f"  Best val loss: {min(history.history['val_total_loss']):.4f} at epoch {history.history['val_total_loss'].index(min(history.history['val_total_loss'])) + 1}")

print("\n─── Applying EMA weights to generator ───")
cvae.apply_ema_weights()

print("\n─── Per-class reconstruction quality (EMA) ───")
for c in range(NUM_CLASSES):
    mask = y_test == c
    class_images = x_test[mask][:100]
    class_labels = tf.one_hot(np.full(len(class_images), c), NUM_CLASSES)
    z_mean, z_log_var, z = cvae.encoder([class_images, class_labels])
    recon = cvae.generator([z, class_labels])
    mse = tf.reduce_mean(tf.square(class_images - recon)).numpy()
    print(f"  Class {c}: MSE={mse:.6f}")

save_dir = os.path.dirname(os.path.abspath(__file__))
encoder.save(os.path.join(save_dir, "cvae_encoder.h5"))
generator.save(os.path.join(save_dir, "cvae_generator.h5"))
print(f"\nModels saved to {save_dir}")
