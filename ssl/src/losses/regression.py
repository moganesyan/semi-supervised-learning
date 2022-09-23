import tensorflow as tf


def pi_model_se(y_pred_1: tf.Tensor,
                y_pred_2: tf.Tensor) -> tf.Tensor:
    """
        Scaled squared error (SE) loss.
        As defined in the Pi Model original paper: https://arxiv.org/abs/1610.02242

        - Used to calculate the unsupervised loss component.
        - Scaled by the number of classes.
        - Operates on a softmax output vector.

        args:
            y_pred_1: (tf.Tensor) - Output predictions under first augmentation realisation.
            y_pred_2: (tf.Tensor) - Output predictions under second augmentation realisation.
        returns:
            loss (tf.Tensor): Scaled SE loss.
    """

    _epsilon = tf.constant(1e-16)
    num_classes = tf.shape(y_pred_2)[1]

    mse = tf.reduce_sum(
        tf.math.pow(y_pred_1 - y_pred_2, 2),
        axis = -1
    )

    return tf.reduce_mean(mse / num_classes)
