#!/usr/bin/env python3
"""Print one generated CVAE image for class 3 as ASCII art."""

import numpy as np
import onnxruntime as ort

from pathlib import Path

save_dir = Path(__file__).resolve().parent
gen_session = ort.InferenceSession(str(save_dir / "cvae_generator.onnx"))
gen_input_names = [inp.name for inp in gen_session.get_inputs()]
gen_output_name = gen_session.get_outputs()[0].name

label_input_name = next(n for n in gen_input_names if 'label_input' in n)
latent_input_name = next(n for n in gen_input_names if 'latent_input' in n)

label_info = gen_session.get_inputs()[gen_input_names.index(label_input_name)]
latent_info = gen_session.get_inputs()[gen_input_names.index(latent_input_name)]
NUM_CLASSES = int(label_info.shape[1])
LATENT_DIM = int(latent_info.shape[1])

label_oh = np.zeros((1, NUM_CLASSES), dtype=np.float32)
label_oh[0, 4] = 1.0
noise = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)

gen_inputs = {latent_input_name: noise, label_input_name: label_oh}
gen_images = gen_session.run([gen_output_name], gen_inputs)[0][0, :, :, 0]

if gen_images.max() > 1.0:
    gen_images = gen_images / 255.0

img = (gen_images > 0.5).astype(np.float32)
img = gen_images

chars = " .:-=+*#%@"
for row in img:
    print("".join(chars[int(val * (len(chars) - 1))] for val in row))

print("---")

#label_oh = np.zeros((1, NUM_CLASSES), dtype=np.float32)
#label_oh[0, 1] = 1.0
#noise = np.random.normal(size=(1, LATENT_DIM)).astype(np.float32)
#
#gen_inputs = {latent_input_name: noise, label_input_name: label_oh}
#gen_images = gen_session.run([gen_output_name], gen_inputs)[0][0, :, :, 0]
#
#if gen_images.max() > 1.0:
#    gen_images = gen_images / 255.0
#
#img = (gen_images > 0.5).astype(np.float32)
#
#chars = " .:-=+*#%@"
#for row in img:
#    print("".join(chars[int(val * (len(chars) - 1))] for val in row))
