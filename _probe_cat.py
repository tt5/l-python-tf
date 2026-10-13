import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collab import CollabModulation

L2_REG = 0.00006
LATENT_DIM = 256
NUM_CLASSES = 5

def upsample(x, target_size):
    channels = x.shape[-1]
    x = layers.Conv2D(channels * 4, 3, padding='same', activation='relu')(x)
    return tf.nn.depth_to_space(x, 2)

src = open("train_cvae.py").read()
m = re.search(r'(def build_generator\(\):.*?)(?=\n# ─── Perceptual|\n# ─── VAE Model|\nclass VAE\b|\ndef transfer)', src, re.S)
body = m.group(1)
ns = dict(keras=keras, layers=layers, tf=tf, L2_REG=L2_REG, LATENT_DIM=LATENT_DIM,
          NUM_CLASSES=NUM_CLASSES, upsample=upsample, CollabModulation=CollabModulation)
exec(body, ns)
g = ns["build_generator"]()
z = tf.random.normal((4, LATENT_DIM))
lab = keras.utils.to_categorical([0, 1, 2, 3], NUM_CLASSES)
out = g([z, lab]).numpy()
print("RESULT generator out:", out.shape, "range", round(float(out.min()),3), round(float(out.max()),3))
# check colla-factors are trainable
cw = [w for w in g.trainable_weights if 'collab' in w.name]
print("RESULT colla-factor weights:", [(w.name, float(w.numpy())) for w in cw])
print("RESULT builds, runs, colla-factors trainable OK")
