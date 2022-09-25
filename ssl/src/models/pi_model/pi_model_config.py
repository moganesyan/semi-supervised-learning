from typing import Union, Tuple, List, Dict

from ..base_model.base_model_config import BaseModelConfig

from tensorflow.keras.layers import ReLU, Softmax, Layer


class PiModelConfig(BaseModelConfig):
    """
        PiModel configuration class.

        As seen in the original paper: https://arxiv.org/abs/1610.02242

        params:
            input_shape (Tuple[int]) - Input tensor shape.
            output_shape (Union[int, Tuple[int]]) - Output tensor shape.
            architecture (Dict) - Architecture config object.
                Defined according to 'block' structure.
    """

    input_shape: Tuple[int, int, int] = (32, 32, 3)
    output_shape: Union[int, Tuple[int]] = 10
    output_activation: Layer = Softmax

    architecture: List[Dict] = [
        {'name': 'conv1', 'type': 'conv', 'repeats': 3, 'filters': 128, 'k': 3, 's': 2, 'pad': 'same', 'act':  ReLU},
        {'name': 'drop1', 'type': 'drop', 'ratio': 0.50},
        {'name': 'conv2', 'type': 'conv', 'repeats': 3, 'filters': 256, 'k': 3, 's': 2, 'pad': 'same', 'act': ReLU},
        {'name': 'drop2', 'type': 'drop', 'ratio': 0.50},
        {'name': 'conv3', 'type': 'conv', 'repeats': 1, 'filters': 512, 'k': 3, 's': 1, 'pad': 'valid', 'act': ReLU},
        {'name': 'conv4', 'type': 'conv', 'repeats': 1, 'filters': 256, 'k': 1, 's': 1, 'pad': 'valid', 'act': ReLU},
        {'name': 'conv5', 'type': 'conv', 'repeats': 1, 'filters': 128, 'k': 1, 's': 1, 'pad': 'valid', 'act': ReLU},
        {'name': 'gpool1', 'type': 'g_avg_pool'},
        {'name': 'fc1', 'type': 'dense', 'repeats': 1, 'units': 128, 'act': ReLU}
    ]
