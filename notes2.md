# Notes

project: python-tf

goal:  >50% correct generation/classification for every class as measured by test_cvae.py

## dependencies

- synthetic project (../../synthetic)
- telephonegame project (../../telephonegame)
- onnx runtime compatible, convert_generator.py should not change (because the synthetic project depends on it)

## models

training:

mnist.py (data: synthetic), mnist2.py (data: telephonegame/data/pbm), mnist3.py (data: telephonegame/data/pbm2)
train_cvae.py, train_cvae2.py, train_cvae3.py

interference:

mnist_model.onnx, mnist2_model.onnx, mnist3_model.onnx
