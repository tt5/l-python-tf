# Best Model Log

## mnist2.py Training

- **Saved model path:** `mnist2_model` (SavedModel directory)
- **Best criteria:** EarlyStopping monitored `val_loss` with `restore_best_weights=True`
- **Saved model mtime:** 2026-07-07T03:21:32
- **Latest TensorBoard validation events:** `logs/mnist2b/validation/events.out.tfevents.1783386961.debian.691848.1.v2`

## Related artifacts in repo

- `best_generator.h5` — 33,670,952 bytes, mtime 2026-07-06T04:23:05
- `best_model.weights.h5` — 45,372,208 bytes, mtime 2026-06-25T00:58:50
- `best_recon.weights.h5` — 5,015,376 bytes, mtime 2026-07-04T05:20:07

Note: `mnist2.py` does not create the `best_*.h5` files; those come from other scripts/trainers.
