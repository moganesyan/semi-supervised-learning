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

    mse = tf.reduce_sum(
        tf.math.pow(y_pred_1 - y_pred_2, 2),
        axis = -1
    )

    return tf.reduce_mean(mse)


def masked_consistency(y_pred_1: tf.Tensor,
                       y_pred_2: tf.Tensor,
                       mask: tf.Tensor) -> tf.Tensor:
    """
        Masked consistency loss.

        - Used to calculate the unsupervised loss components.
        - Mask used to zero out undesired samples.
        - Operates on a softmax output vector.

        args:
            y_pred_1: (tf.Tensor) - Output predictions under first augmentation realisation.
            y_pred_2: (tf.Tensor) - Output predictions under second augmentation realisation.
            mask (tf.Tensor): Mask vector for "switching off" loss
                for certain samples.
        returns:
            loss (tf.Tensor): Scaled SE loss.
    """

    mask_num = tf.cast(mask, tf.float32)
    _epsilon = tf.constant(1e-16)

    mse = tf.reduce_sum(
        tf.math.pow(y_pred_1 - y_pred_2, 2),
        axis = -1
    )
    mse_masked = mse[..., tf.newaxis] * mask_num

    # calculate modified batchwise mean
    minibatch_size = tf.reduce_sum(mask_num)

    return tf.reduce_sum(mse_masked) / (minibatch_size + _epsilon)
