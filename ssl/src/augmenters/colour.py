from collections import namedtuple
from typing import Tuple

import tensorflow as tf


def colour_jitter(x_in: tf.Tensor, strength: float) -> tf.Tensor:
    """
        Apply colour jitter.
        1) Tweak brightness.
        2) Tweak contrast.
        3) Tweak saturation.
        4) Tweak hue.
        args:
            x_in (tf.Tensor): Input image tensor.
            strength (float): Strength of colour distortion.
        returns:
            x_out (tf.Tensor): Augmented image tensor.
    """

    x = tf.image.random_brightness(x_in, max_delta=0.8 * strength)
    x = tf.image.random_contrast(x, lower=1-0.8 * strength, upper=1+0.8 * strength)
    if x_in.shape[-1] == 3:
        x = tf.image.random_saturation(x, lower=1-0.8 * strength, upper=1+0.8 * strength)
        x = tf.image.random_hue(x, max_delta=0.2 * strength)
    x_out = tf.clip_by_value(x, 0, 1)

    return x_out


def colour_drop(x_in: tf.Tensor) -> tf.Tensor:
    """
        Apply colour jitter.
        1) Convert to grayscale.
        2) Reconvert to RGB.
        args:
            x_in (tf.Tensor): Input image tensor.
        returns:
            x_out (tf.tensor): Augmented image tensor.
    """

    x = tf.image.rgb_to_grayscale(x_in)
    x_out = tf.tile(x, [1, 1, 1, 3])

    return x_out


def apply_colour_distortion(x_in: tf.Tensor,
                            distort_strength: float = 0.50,
                            **kwargs) -> tf.Tensor:
    """
        Apply colour distortion augmentations.
        1) Apply random colour jitter.
        2) Apply random colour drop.
        args:
            x_in (tf.Tensor): Input image tensor.
            distort_strength (float): Strength of colour distortion.
        returns:
            x_out (tf.Tensor): Augmented image tensor.
    """

    apply_jitter = tf.random.uniform(
        (), minval = 0, maxval = 1.0, dtype = tf.float32)
    apply_drop = tf.random.uniform(
        (), minval = 0, maxval = 1.0, dtype = tf.float32)

    x_out = x_in
    if apply_jitter <= 0.80:
        x_out = colour_jitter(x_out, distort_strength)
    if x_in.shape[-1] == 3:
        if apply_drop <= 0.20:
            x_out = colour_drop(x_out)

    return x_out
