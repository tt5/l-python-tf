"""collab.py

CAT-style learned collaboration between FiLM (channel) modulation and
spatial attention, for the conditional generator.

    x' = x * (c_w * gamma + s_w * att) + beta

c_w, s_w are trainable colla-factors (CAT, arXiv 2212.06335). Init
c_w=1, s_w=0 so the module starts as pure FiLM and learns how much
spatial attention to add.
"""

from tensorflow.keras import layers


class CollabModulation(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.c_w = self.add_weight(name="c_w", shape=(),
                                   initializer="ones", trainable=True)
        self.s_w = self.add_weight(name="s_w", shape=(),
                                   initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        x, gamma, att, beta = inputs
        modulation = self.c_w * gamma + self.s_w * att
        return x * modulation + beta

    def get_config(self):
        config = super().get_config()
        return config
