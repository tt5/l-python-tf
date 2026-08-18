"""Stage 2/3 PBM-only data loader.

Loads P4 PBM images from ``class_<c>`` directories, upscales to ``image_shape``,
binarizes, splits into train/test, and optionally applies per-class sampling.
No .npy loading or synthetic class generation.
"""

from pathlib import Path

import numpy as np


def _load_pbm(path):
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
            grid[y, x] = 1.0 - bit  # invert: P4 black=1 -> white digit=1
    return grid


def _upscale_nearest(grid, dst_h, dst_w):
    """Nearest-neighbor block upscale from `grid` to (dst_h, dst_w)."""
    src_h, src_w = grid.shape
    out = np.zeros((dst_h, dst_w), dtype=np.float32)
    base_h, extra_h = divmod(dst_h, src_h)
    base_w, extra_w = divmod(dst_w, src_w)
    for r in range(src_h):
        r_start = r * base_h + min(r, extra_h)
        r_end = min(r_start + base_h + (1 if r < extra_h else 0), dst_h)
        for c in range(src_w):
            c_start = c * base_w + min(c, extra_w)
            c_end = min(c_start + base_w + (1 if c < extra_w else 0), dst_w)
            out[r_start:r_end, c_start:c_end] = grid[r, c]
    return out


def load_classes(pbm_dir, num_classes, image_shape=(28, 28, 1), max_samples_per_class=None, split_ratio=0.8):
    """Load PBM images from ``pbm_dir/class_<c>/*.pbm``.

    Parameters
    ----------
    pbm_dir : pathlib.Path
        Root directory containing ``class_0/``, ``class_1/``, ... subdirectories.
    num_classes : int
        Number of classes to load.
    image_shape : tuple[int, int, int]
        Expected output shape ``(H, W, C)``.
    max_samples_per_class : int or None
        If set, load at most this many samples per class before splitting.
    split_ratio : float
        Fraction of each class to use for training.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        Numpy arrays ready for model training/evaluation.
    """
    h, w, _ = image_shape
    x_train, y_train, x_test, y_test = [], [], [], []

    for c in range(num_classes):
        files = sorted((pbm_dir / f"class_{c}").glob("*.pbm"))
        if not files:
            raise ValueError(f"No .pbm files in {pbm_dir / f'class_{c}'}")
        if max_samples_per_class is not None and max_samples_per_class > 0:
            files = files[:max_samples_per_class]
        split_idx = int(len(files) * split_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        for f in train_files:
            img = _load_pbm(f)
            if img.shape != (h, w):
                img = _upscale_nearest(img, h, w)
            img = img.reshape(image_shape)
            img = (img > 0.5).astype(np.float32)
            x_train.append(img)
            y_train.append(c)
        for f in test_files:
            img = _load_pbm(f)
            if img.shape != (h, w):
                img = _upscale_nearest(img, h, w)
            img = img.reshape(image_shape)
            img = (img > 0.5).astype(np.float32)
            x_test.append(img)
            y_test.append(c)

        print(f"Class {c}: {len(train_files)} train, {len(test_files)} test")

    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    train_perm = np.random.permutation(len(x_train))
    test_perm = np.random.permutation(len(x_test))
    x_train, y_train = x_train[train_perm], y_train[train_perm]
    x_test, y_test = x_test[test_perm], y_test[test_perm]

    return (x_train, y_train), (x_test, y_test)


__all__ = ["load_classes"]
