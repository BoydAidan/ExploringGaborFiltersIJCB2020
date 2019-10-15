from keras import backend as K
from keras.layers import Layer
import numpy as np
import tensorflow as tf

class BitSampling(Layer):

    def __init__(self, out_dim, **kwargs):
        self.output_dim = out_dim
        super(BitSampling, self).__init__(**kwargs)


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # self.kernel = self.add_weight(name='kernel',
        #                               shape=(input_shape[1], self.output_dim),
        #                               initializer='uniform',
        #                               trainable=True)
        super(BitSampling, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
		# Extracts specific bits
        map = x[:, 6:49:6, 7::16, :]
        map = K.reshape(map, (-1, 1536))

        return map

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
