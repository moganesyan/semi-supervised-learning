from collections import namedtuple
from typing import Tuple

import tensorflow as tf


def apply_crop_and_resize(x_in: tf.Tensor,
                           crop_size: Tuple[float, float] = (0.08, 1.0),
                           aspect_range: Tuple[float, float] = (0.75, 1.33),
                           num_tries: int = 100,
                           **kwargs) -> tf.Tensor:
    """
        Random crop and resize based on crop size and aspect ratio ranges.
            1) Sample crop size and aspect ratio.
            2) Get crop dimensions.
            3) Adjust crop dimensions to aspect ratio.
            3) Check that the crop dimensions are valid.
            4) Crop image based on valid crop dimensions and resize to original dimensions.
            5) Return original image if valid crop can't be generated within num_tries.
        args:
            x_in (tf.Tensor): Input image tensor.
            crop_size (Tuple[float, float]): Crop size range (proprtion of input image).
            aspect_range (Tuple[float, float]): Aspect ratio range.
            num_tries (int): Number of tries to generate crop within given constraints.
        returns:
            x_out (tf.Tensor):Cropped image tensor.
    """

    h_original = x_in.shape[1]
    w_original = x_in.shape[2]
    ch_original = x_in.shape[3]

    resize_dims = [h_original, w_original]

    crop_size_min = crop_size[0]
    crop_size_max = crop_size[1]

    aspect_ratio_min = aspect_range[0]
    aspect_ratio_max = aspect_range[1]

    # initialise tf loop variables
    tf_counter = tf.constant(0)
    stop_flag = tf.constant(0)
    x_out = x_in

    input_pair = namedtuple('input_pair', 'x_out, stop_flag')
    loop_vars = [tf_counter, input_pair(x_out, stop_flag)]
    shape_invariants = [
        tf_counter.get_shape(),
        input_pair(tf.TensorShape([None, h_original, w_original, ch_original]),
        stop_flag.get_shape())
    ]

    # define operation block
    def block(x_in, stop_flag):
        crop_resized = x_in

        # randomly get crop area and aspect ratio
        crop_size = tf.random.uniform(
            (), minval = crop_size_min, maxval = crop_size_max)
        aspect_ratio = tf.random.uniform(
            (), minval = aspect_ratio_min, maxval = aspect_ratio_max)

        # calculate the desired height and width of crop based on crop size
        num_pixels_original = h_original * w_original
        num_pixels_new = tf.math.floor(num_pixels_original * crop_size)

        w_new = tf.math.floor(tf.math.sqrt(aspect_ratio * num_pixels_new))
        h_new = tf.math.floor(num_pixels_new / w_new)

        h_new = tf.cast(h_new, tf.int64)
        w_new = tf.cast(w_new, tf.int64)

        if w_new <= w_original and h_new <= h_original:
            crop_dims = tf.stack(
                (tf.shape(x_in)[0], h_new, w_new, ch_original),
                axis = 0
            )
            crop = tf.image.random_crop(x_in, crop_dims)
            crop_resized = tf.image.resize(crop, resize_dims)
            stop_flag = tf.constant(1)

        return input_pair(crop_resized, stop_flag)

    output_payload = tf.while_loop(
        lambda tf_counter, p: tf_counter < num_tries and p.stop_flag == 0,
        lambda tf_counter, p: [tf_counter + 1, block(p.x_out, p.stop_flag)],
        loop_vars = loop_vars,
        shape_invariants = shape_invariants
    )
    return output_payload[1].x_out