#!/usr/bin/env python3
"""convert_generator.py

Convert the CVAE generator to ONNX format.
Handles custom layers like LayerNormalization.

Usage:
    python convert_generator.py
"""

import tensorflow as tf

custom_objects = {
    "LayerNormalization": tf.keras.layers.LayerNormalization,
}

model = tf.keras.models.load_model("cvae_generator.h5", custom_objects=custom_objects)

# Save as SavedModel first, then convert
model.save("cvae_generator_savedmodel")

# Use tf2onnx to convert
import subprocess
subprocess.run([
    "python", "-m", "tf2onnx.convert",
    "--saved-model", "cvae_generator_savedmodel",
    "--output", "cvae_generator.onnx",
], check=True)

print("Conversion complete!")
