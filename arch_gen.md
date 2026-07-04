# stage 3 generator

## current

early stopping
data augmentation
dropout
lr schedule
residual connections

Contrastive z regularization Within each batch, push apart latents of different classes, pull together latents of the same class. Straightforward cosine/InfoNCE loss on z. Often helps when the bottleneck is class separation, not reconstruction.

Cross-attention: flatten spatial features, use label embedding as query, attend over spatial keys/values; 1–2 heads enough for 7×7/14×14

Multi-scale kernel block: parallel 1×1, 3×3, 5×5 convs concatenated, then 1×1 reduction

---

FiLM: per-channel scale+shift from label embedding, like your train_cvae2.py film/film_enc

Label embedding tower: separate MLP for label before fusion, rather than raw one-hot

gradient clipping

SE block: squeeze-and-excitation on encoder spatial features before flatten

Perceptual loss: reuse mnist3_model features conv14 and conv7_deep2 as perceptual extractors

---

Latent post-projection: after sampling z, pass through small Dense→LayerNorm→Act before decoder

Dual-path conditioning: one path processes image, one path processes label, fuse by concat at multiple scales

Residual blocks: replace plain conv stacks with pre-activation ResNet blocks (Conv→Norm→Act→Conv→Add)

Conditional Instance Norm / AdaIN: normalize features, then label-predicted affine

U-Net skip connections: expose intermediate encoder spatial features (7×7, 14×14), project and add to decoder at corresponding scales

Classifier-guided generation at inference Use your frozen mnist3 classifier to rank candidate latent samples for each class, then accept only the best-scoring latent. No model architecture change, only eval/post-processing. Cheap to try.

Classifier-guided latent search — for each target class, sample many z, score through frozen mnist3_model, keep top latent

Multi-sample generation / voting — generate N images per label, pick best by classifier confidence


