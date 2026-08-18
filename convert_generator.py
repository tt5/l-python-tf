#!/usr/bin/env python3
"""convert_generator.py

Convert the CVAE generator to ONNX format.
Handles custom layers like LayerNormalization.

Usage:
    python convert_generator.py
"""

import argparse
import os

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="cvae_generator.onnx", help="Output ONNX file path")
parser.add_argument("--model", default="cvae_generator.h5", help="Input Keras model path")
args = parser.parse_args()

custom_objects = {
    "LayerNormalization": tf.keras.layers.LayerNormalization,
}

model = tf.keras.models.load_model(args.model, custom_objects=custom_objects)

# Save as SavedModel first, then convert
savedmodel_dir = os.path.splitext(args.model)[0] + "_savedmodel"
model.save(savedmodel_dir)

# Use tf2onnx to convert
import subprocess
subprocess.run([
    "python", "-m", "tf2onnx.convert",
    "--saved-model", savedmodel_dir,
    "--output", args.output,
], check=True)

print(f"Conversion complete: {args.output}")
