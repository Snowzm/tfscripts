# code by snow 2/10/2019
# mengxue_zhang@outlook.com

# usage:
# Spectrum_Pooling(out_size=42)
# Spectrum_Pooling(pool_size=(2, 2))

import tensorflow as tf
from keras.layers import Layer
import math
from fft2d import fft2, fft_shift


# Reference: Spectral Representations for Convolutional Neural Networks
class SpectralPooling(Layer):
    # default style is 'channels_last'
    def __init__(self, pool_size=(2, 2), out_size=None, **kwargs):
        self.ps = pool_size
        self.out_size = out_size
        super(SpectralPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        if not self.out_size:
            self.out_size = int(input_shape[1] / self.ps[0])

        mid = math.ceil((input_shape[1] - 1) / 2)
        self.b = mid - math.floor(self.out_size / 2)
        self.e = self.b + self.out_size

    def truncate(self, x):
        def tf_real(complex_x):
            x = tf.real(complex_x)
            return tf.complex(real=x, imag=tf.zeros_like(x))

        x = fft_shift(x)
        trun_x = x[:, self.b:self.e, self.b:self.e, :]

        # constraints
        if self.out_size % 2 == 0:
            c = self.out_size // 2
            c_c = tf_real(tf.expand_dims(trun_x[:, c, c, :], axis=1))
            cr = tf.expand_dims(tf.concat([trun_x[:, c, 0:c, :], c_c, trun_x[:, c, c + 1:, :]], axis=1), axis=1)
            trun_x = tf.concat([tf.expand_dims(tf.zeros_like(trun_x[:, 0, :, :]), axis=1), trun_x[:, 1:c, :, :], cr,
                                trun_x[:, c + 1:, :, :]], axis=1)
            trun_x = tf.concat([tf.expand_dims(tf.zeros_like(trun_x[:, :, 0, :]), axis=2), trun_x[:, :, 1:, :]], axis=2)

        trun_x = fft_shift(trun_x, True)
        return trun_x

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.out_size, self.out_size, input_shape[3]

    def call(self, x, mask=None):
        fft_x = fft2d(x, True)
        trun_x = self.truncate(fft_x)
        x = fft2d(trun_x, False)
        img_max = tf.reduce_max(tf.abs(tf.imag(x)))
        with tf.control_dependencies([tf.assert_less(img_max, 1e-5)]):
            return tf.real(x)
