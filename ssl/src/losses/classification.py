import tensorflow as tf


def categorical_cross_entropy(y_pred: tf.Tensor, y_test: tf.Tensor) -> tf.Tensor:
    """
        Categorical cross-entropy loss.

        args:
            y_pred (tf.Tensor): Output prediction vector.
            y_test (tf.Tensor): Ground truth vector.
        returns:
            loss (tf.Tensor): Categorical CE loss.
    """

    cce = tf.reduce_sum(
        y_test * tf.math.log(tf.clip_by_value(y_pred, 1e-16, 1e16)),
        axis = -1
    )
    return tf.reduce_mean(-cce)


def categorical_cross_entropy_masked(y_pred: tf.Tensor,
                                     y_test: tf.Tensor,
                                     mask: tf.Tensor) -> tf.Tensor:
    """
        Masked categorical cross-entropy loss.

        args:
            y_pred (tf.Tensor): Output prediction vector.
            y_test (tf.Tensor): Ground truth vector.
            mask (tf.Tensor): Mask vector for "switching off" loss
                for certain samples.
        returns:
            loss (tf.Tensor): Categorical CE loss.
    """

    mask_num = tf.stop_gradient(tf.cast(mask, tf.float32))

    cce = -1 * tf.reduce_sum(
        y_test * tf.math.log(tf.clip_by_value(y_pred, 1e-16, 1e16)),
        axis = -1
    )

    cce_maksed = cce[..., tf.newaxis] * mask_num

    # calculate modified batchwise mean
    minibatch_size = tf.stop_gradient(
        tf.clip_by_value(tf.reduce_sum(mask_num), 1e-16, 1e16)
    )

    return tf.reduce_sum(cce_maksed) / minibatch_size
