from typing import Union, Tuple, List, Dict

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

from ..base_model.base_model import BaseModel
from .lenet5_config import Lenet5Config


class Lenet5(BaseModel):
    """
        Lenet5 model class.
    """

    def __init__(self, model_config: Lenet5Config) -> None:
        super().__init__(model_config)

        self._input_shape = model_config.input_shape
        self._output_shape = model_config.output_shape
        self._output_activation = model_config.output_activation
        self._architecture = model_config.architecture
    
    def _build_network(self) -> tf.keras.models.Model:
        """
            Build Lenet5 architecture from the `model_config` object.

            args:
                None
            returns:
                model (tf.keras.models.Model) - Keras model callable.
        """
        
        # define input
        x_in = layers.Input(shape = self._input_shape)

        # build architecture
        for block_idx, block in enumerate(self._architecture):
            if block_idx == 0:
                x = x_in
            
            if block['type'] == "conv":
                x = layers.Conv2D(
                    filters = block['filters'],
                    kernel_size = block['k'],
                    strides = block['s'],
                    padding = block['pad'],
                    name = block['name']
                )(x)
                if block['act'] is not None:
                    x = block['act']()(x)
            elif block['type'] == "flat":
                x = layers.Flatten()(x)
            elif block['type'] == "dense":
                x  = layers.Dense(block["units"])(x)
                if block['act'] is not None:
                    x = block['act']()(x)
            else:
                raise ValueError(f"Unknown block type: {block['type']}")
        
        # get output layer
        x_out = layers.Dense(self._output_shape)(x)
        x_out = self._output_activation()(x_out)

        # get keras model callable
        model = models.Model(inputs = x_in, outputs = x_out)
        return model
