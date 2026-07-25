"""Shared synthetic-ish MNIST-like loader for black/white synthesized classes.

This module centralizes dataset creation so that both ``mnist.py`` and
``train_cvae.py`` use the exact same loading behavior.
"""

from pathlib import Path

import numpy as np
import tensorflow.keras as keras


def load_data(data_dir, num_classes=6, flip_pixels=2, image_shape=(28, 28, 1), randomized=True, split_ratio=0.8):
    """Load class 0..num_classes-3 from ``data_dir`` and synthesize two extra classes.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory that contains ``class_<c>/`` folders with ``*.npy`` samples.
    num_classes : int
        Total number of classes. The last two classes are synthesized as
        all-black and all-white.
    flip_pixels : int
        Number of distinct pixel positions flipped in each synthesized image.
        0 disables the flip behavior.
    image_shape : tuple[int, int, int]
        Expected image spatial shape including channels.
    randomized : bool
        Apply per-image random pixel flips for the two synthesized classes.
    split_ratio : float
        Fraction of each class to use for training, remainder for testing.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        Numpy arrays ready for model training/evaluation.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    h, w, _ = image_shape
    x_train, y_train, x_test, y_test = [], [], [], []
    max_train, max_test = 0, 0

    for c in range(num_classes - 2):
        files = sorted(data_dir.glob(f"class_{c}/*.npy"))
        if not files:
            raise ValueError(f"No .npy files in {data_dir / f'class_{c}'}")
        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        for f in train_files:
            img = np.load(str(f)).astype(np.float32) / 255.0
            img = np.asarray(img).reshape(image_shape)
            img = (img > 0.5).astype(np.float32)
            x_train.append(img)
            y_train.append(c)
        for f in test_files:
            img = np.load(str(f)).astype(np.float32) / 255.0
            img = np.asarray(img).reshape(image_shape)
            img = (img > 0.5).astype(np.float32)
            x_test.append(img)
            y_test.append(c)

        max_train = max(max_train, len(train_files))
        max_test = max(max_test, len(test_files))
        print(f"Class {c}: {len(train_files)} train, {len(test_files)} test")

    black = np.zeros(image_shape, dtype=np.float32)
    white = np.ones(image_shape, dtype=np.float32)
    pixel_indices = np.arange(h * w)

    for img, label in [(black, num_classes - 2), (white, num_classes - 1)]:
        for _ in range(max_train):
            sample = img
            if randomized and flip_pixels > 0:
                sample = img.copy().reshape(-1)
                flips = np.random.choice(pixel_indices, size=flip_pixels, replace=False)
                sample = sample.copy()
                sample[flips] = 1 - sample[flips]
                sample = sample.reshape(image_shape)
            x_train.append(sample)
            y_train.append(label)
        for _ in range(max_test):
            sample = img
            if randomized and flip_pixels > 0:
                sample = img.copy().reshape(-1)
                flips = np.random.choice(pixel_indices, size=flip_pixels, replace=False)
                sample = sample.copy()
                sample[flips] = 1 - sample[flips]
                sample = sample.reshape(image_shape)
            x_test.append(sample)
            y_test.append(label)

    x_train = np.array(x_train, dtype=np.float32).reshape(-1, *image_shape)
    x_test = np.array(x_test, dtype=np.float32).reshape(-1, *image_shape)
    y_train = np.array(y_train, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    train_perm = np.random.permutation(len(x_train))
    test_perm = np.random.permutation(len(x_test))
    x_train, y_train = x_train[train_perm], y_train[train_perm]
    x_test, y_test = x_test[test_perm], y_test[test_perm]

    return (x_train, y_train), (x_test, y_test)


__all__ = ["load_data"]
