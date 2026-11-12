#!/usr/bin/env python3
"""convert_generator.py

Convert the CVAE generator to ONNX format.
Handles custom layers like LayerNormalization.

Also emits a prior-conditioned variant (cvae_prior_generator.onnx) that folds
the learned conditional prior p(z|y) into the graph. Its latent input keeps the
name "latent_input", so consumers that sample N(0, I) and match on that name
keep working unchanged -- the graph applies z = mu_y + sigma_y * eps internally.

Usage:
    python convert_generator.py
    python convert_generator.py --prior cvae_prior.npz
"""

import argparse
import os
import subprocess

import numpy as np
import tensorflow as tf

from collab import CollabModulation

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="cvae_generator.onnx", help="Output ONNX file path")
parser.add_argument("--model", default="cvae_generator.h5", help="Input Keras model path")
parser.add_argument("--prior", default="cvae_prior.npz", help="Learned conditional prior from train_cvae.py")
parser.add_argument("--prior-output", default="cvae_prior_generator.onnx", help="Output ONNX for the prior-conditioned generator")
args = parser.parse_args()

custom_objects = {
    "LayerNormalization": tf.keras.layers.LayerNormalization,
    "CollabModulation": CollabModulation,
}

model = tf.keras.models.load_model(args.model, custom_objects=custom_objects)

def to_onnx(savedmodel_dir, output_path):
    subprocess.run([
        "python", "-m", "tf2onnx.convert",
        "--saved-model", savedmodel_dir,
        "--output", output_path,
    ], check=True)
    print(f"Conversion complete: {output_path}")


# Save as SavedModel first, then convert
savedmodel_dir = os.path.splitext(args.model)[0] + "_savedmodel"
model.save(savedmodel_dir)
to_onnx(savedmodel_dir, args.output)

# ─── Prior-conditioned generator ──────────────────────────────────
# The plain generator takes z as an input, so a consumer must know what
# distribution to sample it from. Folding the learned p(z|y) in lets consumers
# keep feeding N(0, I) -- which is what they already do.
if not os.path.exists(args.prior):
    print(f"No prior file at {args.prior} -- skipping prior-conditioned export.")
else:
    prior = np.load(args.prior)
    prior_mean = prior["prior_mean"].astype(np.float32)
    prior_log_var = prior["prior_log_var"].astype(np.float32)
    num_classes, latent_dim = prior_mean.shape

    model_inputs = model.input if isinstance(model.input, (list, tuple)) else [model.input]
    label_input = next((t for t in model_inputs if "label" in t.name.lower()), model_inputs[1])
    if int(label_input.shape[-1]) != num_classes:
        raise ValueError(
            f"prior has {num_classes} classes but the generator label input has "
            f"{int(label_input.shape[-1])}"
        )

    # Named "latent_input" deliberately: consumers select the latent input by
    # name and feed standard-normal noise, which is correct now that the graph
    # applies the prior transform itself.
    noise_input = tf.keras.layers.Input(shape=(latent_dim,), name="latent_input")
    prior_mu_layer = tf.keras.layers.Dense(
        latent_dim, use_bias=False, trainable=False, name="prior_mu_table")
    prior_std_layer = tf.keras.layers.Dense(
        latent_dim, use_bias=False, trainable=False, name="prior_std_table")
    prior_mu = prior_mu_layer(label_input)
    prior_std = prior_std_layer(label_input)
    prior_mu_layer.set_weights([prior_mean])
    prior_std_layer.set_weights([np.exp(0.5 * prior_log_var).astype(np.float32)])

    prior_z = tf.keras.layers.Add(name="prior_z")([
        prior_mu,
        tf.keras.layers.Multiply(name="prior_scale")([prior_std, noise_input]),
    ])
    prior_model = tf.keras.Model(
        [noise_input, label_input],
        model([prior_z, label_input]),
        name="prior_generator",
    )

    prior_savedmodel_dir = os.path.splitext(args.prior_output)[0] + "_savedmodel"
    prior_model.save(prior_savedmodel_dir)
    to_onnx(prior_savedmodel_dir, args.prior_output)
