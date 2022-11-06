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

    _epsilon = tf.constant(1e-16)

    cce = tf.reduce_sum(
        y_test * tf.math.log(y_pred + _epsilon),
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

    mask_num = tf.cast(mask, tf.float32)
    _epsilon = tf.constant(1e-16)

    cce = -1 * tf.reduce_sum(
        y_test * tf.math.log(y_pred + _epsilon),
        axis = -1
    )
    cce_maksed = cce * mask_num

    # calculate modified batchwise mean
    minibatch_size = tf.reduce_sum(mask_num)
    return tf.reduce_sum(cce) / (minibatch_size + _epsilon)
