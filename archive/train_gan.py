#!/usr/bin/env python3
"""train_cvae.py (rewritten: Conditional VAE -> Conditional GAN)

Conditional GAN for synthetic pictogram data.

The previous Conditional VAE's KL term smoothed structured output into
fragmented/soft images -- its core weakness for clean geometric
primitives. A cGAN's adversarial loss produces sharp, committed,
connected output while keeping single-pass fast sampling (no diffusion).

The generator is unchanged from the VAE version: (z, label) -> 28x28,
uniform 64-channel, sub-pixel upsampling, FiLM conditioning, per-class
output scale/shift. This keeps the convert_generator.py / test_cvae.py /
onnx_ascii.py contracts intact (saved as cvae_generator.h5).

Usage:
    /home/n/miniconda3/envs/tf/bin/python train_cvae.py [--num-classes 5]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils.mnist_loader import load_data

# ─── Config ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "synthetic" / "data"
LATENT_DIM = 256
EPOCHS = 100
BATCH_SIZE = 256
NUM_CLASSES = 5

# GAN hyperparameters (starting values; tune from the loss curves)
GEN_LR = 0.0002
DISC_LR = 0.0002
BETA_1 = 0.5
ADV_LAMBDA = 1.0        # weight on adversarial loss
L1_LAMBDA = 100.0       # weight on L1 reconstruction (grounding + stability)
L2_REG = 0.00006
GEN_SPATIAL_DROPOUT = 0.1  # light; heavy decoder dropout hurt line learning
LOG_DIR = "logs/cgan"

parser = argparse.ArgumentParser()
parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
parser.add_argument("--max-samples-per-class", type=int, default=None,
                    help="Maximum samples per class to load from disk")
parser.add_argument("--samples-per-epoch-per-class", type=int, default=None,
                    help="Random subset size per class each epoch")
parser.add_argument("--pbm-dir", type=str, default=None,
                    help="Load P4 PBM images from this directory instead of .npy")
args = parser.parse_args()

NUM_CLASSES = args.num_classes
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
MAX_SAMPLES_PER_CLASS = args.max_samples_per_class
SAMPLES_PER_EPOCH_PER_CLASS = args.samples_per_epoch_per_class
PBM_DIR = Path(args.pbm_dir) if args.pbm_dir else None


# ─── Data ──────────────────────────────────────────────────────────
def load_data_cgan():
    (x_train, y_train), (x_test, y_test) = load_data(
        DATA_DIR, NUM_CLASSES, flip_pixels=7 * 28,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS, pbm_dir=PBM_DIR)
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data_cgan()
print(f"Train: {len(x_train)}, Test: {len(x_test)}")


# ─── Upsample via Sub-pixel Convolution ────────────────────────────
def upsample(x, target_size):
    """Upsample using sub-pixel convolution (depth_to_space)."""
    channels = x.shape[-1]
    x = layers.Conv2D(channels * 4, 3, padding='same', activation='relu')(x)
    x = tf.nn.depth_to_space(x, 2)
    return x


# ─── Generator (unchanged architecture from the VAE version) ───────
def build_generator():
    z = keras.Input(shape=(LATENT_DIM,), name="latent_input")
    lbl = keras.Input(shape=(NUM_CLASSES,), name="label_input")

    def make_film(channels):
        def film_fn(label):
            gamma = layers.Dense(channels, activation="sigmoid")(label)
            beta = layers.Dense(channels)(label)
            return gamma, beta
        return film_fn

    # Seed at 14x14x64: spatial resolution is what lets connected line
    # structure survive; 64 channels is ample. Uniform 64-channel decoder
    # (spatial expansion, not channel expansion, carries the capacity).
    x = layers.Dense(14 * 14 * 64, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(z)
    x = layers.Reshape((14, 14, 64))(x)
    # FiLM at seed resolution
    gamma7, beta7 = make_film(64)(lbl)
    gamma7 = layers.Reshape((1, 1, 64))(gamma7)
    beta7 = layers.Reshape((1, 1, 64))(beta7)
    x = layers.Multiply()([x, gamma7])
    x = layers.Add()([x, beta7])
    # SE attention on initial projection
    se_init = layers.GlobalAveragePooling2D()(x)
    se_init = layers.Dense(64 // 4, activation="relu")(se_init)
    se_init = layers.Dense(64, activation="sigmoid")(se_init)
    se_init = layers.Reshape((1, 1, 64))(se_init)
    x = layers.Multiply()([x, se_init])
    # Spatial attention before conv
    sa_init = layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, sa_init])
    x = layers.SpatialDropout2D(GEN_SPATIAL_DROPOUT)(x)

    # 14x14 block (uniform 64 channels). Label is conditioned once, at the
    # seed above; re-applying FiLM at every adjacent conv was redundant.
    x = layers.Conv2D(64, 3, padding="same", activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    se7 = layers.GlobalAveragePooling2D()(x)
    se7 = layers.Dense(64 // 4, activation="relu")(se7)
    se7 = layers.Dense(64, activation="sigmoid")(se7)
    se7 = layers.Reshape((1, 1, 64))(se7)
    x = layers.Multiply()([x, se7])
    sa7 = layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, sa7])
    x = layers.SpatialDropout2D(GEN_SPATIAL_DROPOUT)(x)
    skip_7 = layers.Conv2D(64, 1, padding="same")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    sa14_pre = layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, sa14_pre])
    x = layers.Add()([x, skip_7])
    x = layers.Activation("relu")(x)

    # 14x14 -> 28x28
    x = layers.Conv2D(64, 3, padding="same", activation="relu",
                      kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(x)
    gamma28, beta28 = make_film(64)(lbl)
    gamma28 = layers.Reshape((1, 1, 64))(gamma28)
    beta28 = layers.Reshape((1, 1, 64))(beta28)
    x = layers.Multiply()([x, gamma28])
    x = layers.Add()([x, beta28])
    se14 = layers.GlobalAveragePooling2D()(x)
    se14 = layers.Dense(64 // 4, activation="relu")(se14)
    se14 = layers.Dense(64, activation="sigmoid")(se14)
    se14 = layers.Reshape((1, 1, 64))(se14)
    x = layers.Multiply()([x, se14])
    sa14 = layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = layers.Multiply()([x, sa14])
    skip_14 = layers.Conv2D(64, 1, padding="same")(x)
    x = upsample(x, [28, 28])
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    skip_14_up = upsample(skip_14, [28, 28])
    x = layers.Add()([x, skip_14_up])
    x = layers.Activation("relu")(x)

    # Per-class output scale+shift: lets each class reach its own output
    # dynamic range (fixed the sparse class-4 range collapse in the VAE).
    logits = layers.Conv2D(1, 3, padding="same", activation=None, dtype="float32")(x)
    out_scale = layers.Dense(1)(lbl)
    out_shift = layers.Dense(1)(lbl)
    out_scale = layers.Reshape((1, 1, 1))(out_scale)
    out_shift = layers.Reshape((1, 1, 1))(out_shift)
    x = layers.Activation("sigmoid")(logits * out_scale + out_shift)
    return keras.Model([z, lbl], x, name="generator")


# ─── Discriminator (conditional critic) ────────────────────────────
def build_discriminator():
    img = keras.Input(shape=(28, 28, 1), name="image_input")
    lbl = keras.Input(shape=(NUM_CLASSES,), name="label_input")

    # Inject the label by broadcasting it to a spatial channel map and
    # concatenating (robust; avoids FiLM in the critic).
    lbl_map = layers.Reshape((1, 1, NUM_CLASSES))(lbl)
    lbl_map = layers.UpSampling2D(size=28, interpolation="nearest")(lbl_map)
    x = layers.Concatenate()([img, lbl_map])

    for filters in (64, 128, 256, 256):
        x = layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(0.2)(x)
    out = layers.Dense(1)(x)  # real/fake logit
    return keras.Model([img, lbl], out, name="discriminator")


# ─── cGAN model with alternating train_step ────────────────────────
class CGAN(keras.Model):
    def __init__(self, generator, discriminator, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.gen_loss_tracker = keras.metrics.Mean(name="gen_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="disc_loss")
        self.gen_adv_tracker = keras.metrics.Mean(name="gen_adv")
        self.gen_l1_tracker = keras.metrics.Mean(name="gen_l1")

    def compile(self, g_opt, d_opt, **kwargs):
        super().compile(**kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker,
                self.gen_adv_tracker, self.gen_l1_tracker]

    def train_step(self, data):
        real_imgs, labels = data
        batch = tf.shape(real_imgs)[0]
        z = tf.random.normal((batch, LATENT_DIM))
        fake_imgs = self.generator([z, labels], training=True)

        # --- train discriminator ---
        with tf.GradientTape() as d_tape:
            d_real = self.discriminator([real_imgs, labels], training=True)
            d_fake = self.discriminator([fake_imgs, labels], training=True)
            d_loss = (
                tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(d_real), logits=d_real))
                + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(d_fake), logits=d_fake)))
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_opt.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))

        # --- train generator ---
        with tf.GradientTape() as g_tape:
            gen_imgs = self.generator([z, labels], training=True)
            d_fake = self.discriminator([gen_imgs, labels], training=True)
            adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(d_fake), logits=d_fake))
            l1_loss = tf.reduce_mean(tf.abs(real_imgs - gen_imgs))
            g_loss = ADV_LAMBDA * adv_loss + L1_LAMBDA * l1_loss
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_opt.apply_gradients(zip(g_grads, self.generator.trainable_weights))

        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        self.gen_adv_tracker.update_state(adv_loss)
        self.gen_l1_tracker.update_state(l1_loss)
        return {m.name: m.result() for m in self.metrics}


# ─── Data sequence ─────────────────────────────────────────────────
class DataSequence(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batch_size, samples_per_epoch_per_class=None):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.samples_per_epoch_per_class = samples_per_epoch_per_class

        if len(labels.shape) == 2 and labels.shape[1] > 1:
            label_indices = np.argmax(labels, axis=1)
        else:
            label_indices = np.array(labels)

        self.class_indices = {}
        for idx, label in enumerate(label_indices):
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
        return self.images[batch_idx], self.labels[batch_idx]

    def on_epoch_end(self):
        self._sample_indices()


# ─── Build + train ─────────────────────────────────────────────────
def main():
    generator = build_generator()
    discriminator = build_discriminator()
    cgan = CGAN(generator, discriminator)
    cgan.compile(
        g_opt=keras.optimizers.Adam(GEN_LR, beta_1=BETA_1),
        d_opt=keras.optimizers.Adam(DISC_LR, beta_1=BETA_1),
    )

    train_seq = DataSequence(x_train, y_train, BATCH_SIZE,
                             samples_per_epoch_per_class=SAMPLES_PER_EPOCH_PER_CLASS)

    best_generator_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "best_generator.h5")

    class GeneratorCheckpoint(keras.callbacks.Callback):
        """Save generator weights when gen_loss improves."""
        def __init__(self, generator, filepath, monitor='gen_loss'):
            super().__init__()
            self.generator = generator
            self.filepath = filepath
            self.monitor = monitor
            self.best = float('inf')

        def on_epoch_end(self, epoch, logs=None):
            current = logs.get(self.monitor)
            if current is not None and current < self.best:
                self.best = current
                self.generator.save_weights(self.filepath)
                print(f"  Saved best generator ({self.monitor}: {current:.4f})")

    callbacks = [
        GeneratorCheckpoint(generator, best_generator_path, monitor='gen_loss'),
        keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
    ]

    cgan.fit(train_seq, epochs=EPOCHS, callbacks=callbacks, verbose=2)

    # Load best generator weights and save for ONNX conversion
    if os.path.exists(best_generator_path):
        generator.load_weights(best_generator_path)
        print(f"\nLoaded best generator weights from {best_generator_path}")

    save_dir = os.path.dirname(os.path.abspath(__file__))
    generator.save(os.path.join(save_dir, "cvae_generator.h5"))
    print(f"\nGenerator saved to {save_dir}/cvae_generator.h5")


if __name__ == "__main__":
    main()
