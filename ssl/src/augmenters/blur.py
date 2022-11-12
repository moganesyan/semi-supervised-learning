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

    x_gauss = tf.math.exp(-(tf.pow(x, 2.0) / (2.0 * tf.pow(sigma, 2.0))))
    x_gauss = x_gauss / tf.math.sqrt((2.0 * 3.14159 * tf.pow(sigma, 2.0)))

    kernel_gauss = tf.tensordot(x_gauss, x_gauss, axes = 0)
    x_scale = tf.reduce_sum(kernel_gauss)

    kernel_gauss = kernel_gauss / x_scale

    return kernel_gauss


def apply_gaussian_blur(x_in: tf.Tensor,
                        kernel_ratio: float = 0.10,
                        blur_strength: Tuple[float, float] = (0.1, 2.0),
                        **kwargs) -> tf.Tensor:
    """
        Apply 2D gaussian blur to input tensor.
        - Uniformly sample blur strength [0.1, 2.0].
        - Kernel size is `kernel_ratio` of the input tensor width.

        args:
            x_in (tf.Tensor): Input tensor.
            kernel_ratio (float): ratio of the input tensor width to set as the blur kernel size.
            blur_strength (Tuple[float, float]): Blur strength range to sample from.
        returns:
            x_out (tf.Tensor): Augmented tensor.
    """

    blur_strength_min = blur_strength[0]
    blur_strength_max = blur_strength[1]

    blur_strength = tf.random.uniform(
        (), blur_strength_min, blur_strength_max)
    kernel_size = tf.cast(x_in.shape[1], tf.float32) * tf.constant(kernel_ratio)
    kernel_size = tf.cast(kernel_size, tf.int64)

    kernel = get_gaussian_kernel(kernel_size, blur_strength)
    kernel = kernel[..., tf.newaxis]
    kernel = tf.tile(
        kernel, tf.constant([1, 1, x_in.shape[-1]]))
    kernel = kernel[..., tf.newaxis]

    x_out = tf.nn.depthwise_conv2d(x_in, kernel, [1,1,1,1], 'SAME')

    return x_out

