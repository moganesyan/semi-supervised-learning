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
