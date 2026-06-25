import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

