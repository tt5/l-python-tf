# stage 3

## current

early stopping
data augmentation
dropout
lr schedule, adaptive gradient lr (batch)
gradient clipping
l2 regulation

residual connections

## classifier

cbam

## cvae

Contrastive z regularization Within each batch, push apart latents of different classes, pull together latents of the same class. Straightforward cosine/InfoNCE loss on z. Often helps when the bottleneck is class separation, not reconstruction.

Multi-scale kernel block: parallel 1×1, 3×3, 5×5 convs concatenated, then 1×1 reduction

Dual-path conditioning: one path processes image, one path processes label, fuse by concat at multiple scales

SE block: squeeze-and-excitation on encoder spatial features before flatten

FiLM: per-channel scale+shift from label embedding

---

Latent post-projection: after sampling z, pass through small Dense→LayerNorm→Act before decoder

Residual blocks: replace plain conv stacks with pre-activation ResNet blocks (Conv→Norm→Act→Conv→Add)

Conditional Instance Norm / AdaIN: normalize features, then label-predicted affine
