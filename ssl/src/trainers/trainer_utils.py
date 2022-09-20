import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    """
        Set random seet for numpy, random and tensorflow packages.

        args:
            seed (int): The seed to be applied
        returns:
            None
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
