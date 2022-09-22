from collections import namedtuple
from typing import Tuple

import tensorflow as tf


def get_gaussian_kernel(k: int, sigma: float) -> tf.Tensor:
    """
        Get kxk 2D gaussian kernel.
        args:
            k (int): Kernel size.
            sigma (float): Blur strength.
        returns:
            kernel_gauss (tf.Tensor): Gaussian kernel tensor.
    """

    x = tf.range(-k // 2 + 1, k // 2 + 1, dtype = tf.float32)

    x_gauss = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(sigma, 2))))
    x_gauss = x_gauss / tf.math.sqrt((2 * 3.14159 * tf.pow(sigma, 2)))

    kernel_gauss = tf.tensordot(x_gauss, x_gauss, axes = 0)
    x_scale = tf.reduce_sum(kernel_gauss)

    kernel_gauss = kernel_gauss / x_scale

    return kernel_gauss


def apply_gaussian_blur(x_in: tf.Tensor,
                        blur_strength: Tuple[float, float] = (0.1, 2.0),
                        **kwargs) -> tf.Tensor:
    """
        Apply 2D gaussian blur to input tensor.
        - Uniformly sample blur strength [0.1, 2.0]
        - Kernel size is 10% of the input tensor height / width

        args:
            x_in (tf.Tensor): Input tensor.
            blur_strength (Tuple[float, float]): Blur strength range to sample from.
        returns:
            x_out (tf.Tensor): Augmented tensor.
    """

    blur_strength_min = blur_strength[0]
    blur_strength_max = blur_strength[1]

    blur_strength = tf.random.uniform(
        (), blur_strength_min, blur_strength_max)
    kernel_size = tf.cast(x_in.shape[1], tf.float32) * tf.constant(0.10)
    kernel_size = tf.cast(kernel_size, tf.int32)

    kernel = get_gaussian_kernel(kernel_size, blur_strength)
    kernel = kernel[..., tf.newaxis]
    kernel = tf.tile(
        kernel, tf.constant([1, 1, x_in.shape[-1]]))
    kernel = kernel[..., tf.newaxis]

    x_out = tf.nn.depthwise_conv2d(x_in, kernel, [1,1,1,1], 'SAME')

    return x_out

