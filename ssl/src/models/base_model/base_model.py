from abc import ABC, abstractmethod

from typing import Tuple, Union

import tensorflow as tf

from .base_model_config import BaseModelConfig


class BaseModel(ABC):
    """
        Base class for all models.

        args:
            model_config (BaseModelConfig) - Model configuration parameters.
        returns:
            None
    """

    def __init__(self,
                 model_config: BaseModelConfig) -> None:

        self._model_config = model_config

    @abstractmethod
    def _build_network(self) -> tf.keras.models.Model:
        """
            Model architecture to be constructed using the `model_config`
            object containing architecture configuration parameters.
            This process is architecture / model specific.

            args:
                None
            returns:
                model (tf.keras.models.Model) - Keras model callable.
        """

        pass

    def __call__(self) -> tf.keras.models.Model:
        """
            Get model callable.

            args:
                None
            returns:
                model (tf.keras.models.Model) - Keras model callable.
        """

        return self._build_network()
