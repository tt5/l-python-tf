# stage 3 generator

## current

early stopping
data augmentation
dropout
lr schedule
residual connections
gradient clipping

Contrastive z regularization Within each batch, push apart latents of different classes, pull together latents of the same class. Straightforward cosine/InfoNCE loss on z. Often helps when the bottleneck is class separation, not reconstruction.

Cross-attention: flatten spatial features, use label embedding as query, attend over spatial keys/values; 1–2 heads enough for 7×7/14×14

Multi-scale kernel block: parallel 1×1, 3×3, 5×5 convs concatenated, then 1×1 reduction

Dual-path conditioning: one path processes image, one path processes label, fuse by concat at multiple scales

SE block: squeeze-and-excitation on encoder spatial features before flatten

FiLM: per-channel scale+shift from label embedding

---

Perceptual loss: reuse mnist3_model

---

Latent post-projection: after sampling z, pass through small Dense→LayerNorm→Act before decoder

Residual blocks: replace plain conv stacks with pre-activation ResNet blocks (Conv→Norm→Act→Conv→Add)

Conditional Instance Norm / AdaIN: normalize features, then label-predicted affine
