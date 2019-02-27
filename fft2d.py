# code by snow 2/10/2019
# mengxue_zhang@outlook.com

import tensorflow as tf
import math


def fft2(x, inverse=False):
    # [b, h, w, c]
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    if inverse:
        x = tf.spectral.ifft2d(x)
    else:
        if x.dtype.is_complex:
            x = tf.spectral.fft2d(x)
        else:
            x = tf.spectral.fft2d(tf.complex(real=x, imag=tf.zeros_like(x)))

    x = tf.transpose(x, perm=[0, 2, 3, 1])
    return x


def fft_shift(x, inverse=False):
    # [b, h, w, c]
    shape = x.get_shape().as_list()
    if inverse:
        h = math.floor(shape[1] / 2)
        w = math.floor(shape[2] / 2)
    else:
        h = math.ceil(shape[1] / 2)
        w = math.ceil(shape[2] / 2)

    x1 = x[:, 0:h, 0:w, :]
    x2 = x[:, 0:h, w - shape[2]:, :]
    x21 = tf.concat([x2, x1], axis=-2)
    x3 = x[:, h - shape[1]:, 0:w, :]
    x4 = x[:, h - shape[1]:, w - shape[2]:, :]
    x43 = tf.concat([x4, x3], axis=-2)

    return tf.concat([x43, x21], axis=-3)
