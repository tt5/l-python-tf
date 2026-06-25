#!/usr/bin/env python3
"""convert_to_onnx.py

Convert a TensorFlow SavedModel to ONNX format.

Usage:
    python convert_to_onnx.py <saved_model_dir> <output_onnx_path>

Example:
    python convert_to_onnx.py /home/n/data/l/nats/mnist_model /home/n/data/l/nats/mnist_model.onnx
"""

import sys
import tensorflow as tf
import tf2onnx


def convert(saved_model_dir: str, output_path: str):
    print(f"Loading SavedModel from: {saved_model_dir}")
    model = tf.keras.models.load_model(saved_model_dir)
    print(f"  Input shape:  {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    # Build input signature from the model's input shape
    # Replace None (batch dim) with None, keep rest as-is
    input_shape = model.input_shape
    if input_shape[0] is None:
        sig_shape = (None,) + input_shape[1:]
    else:
        sig_shape = input_shape

    spec = (tf.TensorSpec(sig_shape, tf.float32, name="input"),)

    print(f"Converting to ONNX...")
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, output_path=output_path
    )
    print(f"Saved ONNX model to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <saved_model_dir> <output_onnx_path>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
