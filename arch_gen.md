# stage 3 generator

## current

Replace Conv2DTranspose with UpSampling2D + Conv2D: avoids checkerboard artifacts and is more ONNX-stable

## ideas

Cross-attention: flatten spatial features, use label embedding as query, attend over spatial keys/values; 1–2 heads enough for 7×7/14×14

Dual-path conditioning: one path processes image, one path processes label, fuse by concat at multiple scales

Class-specific bias injection: learn a per-class bias/offset map added at each decoder scale

Conditional Instance Norm / AdaIN: normalize features, then label-predicted affine

FiLM: per-channel scale+shift from label embedding, like your train_cvae2.py film/film_enc

Residual blocks: replace plain conv stacks with pre-activation ResNet blocks (Conv→Norm→Act→Conv→Add)

U-Net skip connections: expose intermediate encoder spatial features (7×7, 14×14), project and add to decoder at corresponding scales


Hierarchical latent: multiple stochastic layers at different scales

Perceptual loss: reuse mnist3_model features conv14 and conv7_deep2 as perceptual extractors

dropout

gradient clipping

lr schedule

data augmentation

early stopping

SE block: squeeze-and-excitation on encoder spatial features before flatten

Multi-scale kernel block: parallel 1×1, 3×3, 5×5 convs concatenated, then 1×1 reduction

Label embedding tower: separate MLP for label before fusion, rather than raw one-hot

Latent post-projection: after sampling z, pass through small Dense→LayerNorm→Act before decoder

Class-conditional latent alignment loss Add a small classifier head on z trained to predict the class from the latent code. Forces the latent space to cluster by class, which usually fixes misclassified weak classes first. ONNX-safe; only affects encoder training.

Contrastive z regularization Within each batch, push apart latents of different classes, pull together latents of the same class. Straightforward cosine/InfoNCE loss on z. Often helps when the bottleneck is class separation, not reconstruction.

Classifier-guided generation at inference Use your frozen mnist3 classifier to rank candidate latent samples for each class, then accept only the best-scoring latent. No model architecture change, only eval/post-processing. Cheap to try.

InfoVAE / MMD-VAE objective instead of vanilla KL Replace standard KL with MMD or IPM penalty. More robust latent structure for small datasets, and often reduces posterior collapse without KL warmup.

Classifier-guided latent search — for each target class, sample many z, score through frozen mnist3_model, keep top latent

Multi-sample generation / voting — generate N images per label, pick best by classifier confidence


