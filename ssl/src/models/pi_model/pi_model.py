from typing import Union, Tuple, List, Dict

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

from ..base_model.base_model import BaseModel
from .pi_model_config import PiModelConfig


class PiModel(BaseModel):
    """
        PiModel model class.

        As seen in the original paper: https://arxiv.org/abs/1610.02242 
    """

    def __init__(self, model_config: PiModelConfig) -> None:
        super().__init__(model_config)

        self._input_shape = model_config.input_shape
        self._output_shape = model_config.output_shape
        self._output_activation = model_config.output_activation
        self._architecture = model_config.architecture
    
    def _build_network(self) -> tf.keras.models.Model:
        """
            Build PiModel architecture from the `model_config` object.

            args:
                None
            returns:
                model (tf.keras.models.Model) - Keras model callable.
        """
        
        # define input
        x_in = layers.Input(shape = self._input_shape, name = 'input')

        # build architecture
        for block_idx, block in enumerate(self._architecture):
            # handle first block input
            if block_idx == 0:
                x = x_in
            
            # handle convolutional blocks
            if block['type'] == "conv":
                for repeat_idx in range(block['repeats']):
                    active_stride = block['s'] if repeat_idx == block['repeats'] - 1 else 1

                    x = layers.Conv2D(
                        filters = block['filters'],
                        kernel_size = block['k'],
                        strides = active_stride,
                        padding = block['pad'],
                        name = f"{block['name']}_{repeat_idx}"
                    )(x)

                    # activation
                    if block['act'] is not None:
                        x = block['act']()(x)

            # handle flatten blocks
            elif block['type'] == "flat":
                x = layers.Flatten(name = block['name'])(x)

            # handle dropout blocks
            elif block['type'] == "drop":
                x = layers.Dropout(block['ratio'], name = block['name'])(x)

            # handle global avg pooling blocks
            elif block['type'] == "g_avg_pool":
                x = layers.GlobalAveragePooling2D(name = block['name'])(x)

            # handle dense layer blocks
            elif block['type'] == "dense":
                for repeat_idx in range(block['repeats']):
                    x  = layers.Dense(block["units"], name = f"{block['name']}_{repeat_idx}")(x)

                    # activation
                    if block['act'] is not None:
                        x = block['act']()(x)
            else:
                raise ValueError(f"Unknown block type: {block['type']}")
        
        # get output layer
        x_out = layers.Dense(self._output_shape, name = 'output')(x)

        if self._output_activation is not None:
            x_out = self._output_activation()(x_out)

        # get keras model callable
        model = models.Model(inputs = x_in, outputs = x_out)
        return model
