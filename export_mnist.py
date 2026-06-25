#!/usr/bin/env python3
"""export_mnist.py

Export Keras MNIST dataset to CSV format matching EMNIST balanced structure.
Output: data/mnist/mnist-train.csv and data/mnist/mnist-test.csv

Format: label,pixel0,pixel1,...,pixel783 (28x28 = 784 pixels)
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "data" / "mnist"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Train: {len(x_train)}, Test: {len(x_test)}")

# Export train
train_path = OUT_DIR / "mnist-train.csv"
print(f"Writing {train_path}...")
with open(train_path, 'w') as f:
    for i in range(len(x_train)):
        label = y_train[i]
        pixels = x_train[i].flatten()
        line = f"{label}," + ",".join(str(p) for p in pixels)
        f.write(line + "\n")
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(x_train)}")

# Export test
test_path = OUT_DIR / "mnist-test.csv"
print(f"Writing {test_path}...")
with open(test_path, 'w') as f:
    for i in range(len(x_test)):
        label = y_test[i]
        pixels = x_test[i].flatten()
        line = f"{label}," + ",".join(str(p) for p in pixels)
        f.write(line + "\n")
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(x_test)}")

print(f"\nDone. Files:")
print(f"  {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"  {test_path} ({test_path.stat().st_size / 1024 / 1024:.1f} MB)")
