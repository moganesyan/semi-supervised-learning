from typing import Union, Tuple, List, Dict

from ..base_model.base_model_config import BaseModelConfig

from tensorflow.keras.layers import ReLU, Softmax, Layer


class AlexNetConfig(BaseModelConfig):
    """
        AlexNet configuration class.

        Modified from the original paper.

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
        {'name': 'conv1', 'type': 'conv', 'repeats': 1, 'filters': 96, 'k': 11, 's': 1, 'pad': 'same', 'act':  ReLU},
        {'name': 'pool1', 'type': 'max_pool', 'k': 3, 's': 2, 'pad': 'valid'},
        {'name': 'conv2', 'type': 'conv', 'repeats': 1, 'filters': 256, 'k': 5, 's': 1, 'pad': 'same', 'act': ReLU},
        {'name': 'pool2', 'type': 'max_pool', 'k': 3, 's': 2, 'pad': 'valid'},
        {'name': 'conv3', 'type': 'conv', 'repeats': 1, 'filters': 384, 'k': 3, 's': 1, 'pad': 'same', 'act': ReLU},
        # {'name': 'conv4', 'type': 'conv', 'repeats': 1, 'filters': 384, 'k': 3, 's': 1, 'pad': 'same', 'act': ReLU},
        # {'name': 'conv5', 'type': 'conv', 'repeats': 1, 'filters': 256, 'k': 3, 's': 1, 'pad': 'same', 'act': ReLU},
        {'name': 'pool3', 'type': 'max_pool', 'k': 3, 's': 2, 'pad': 'valid'},
        {'name': 'flat1', 'type': 'flat'},
        {'name': 'fc1', 'type': 'dense', 'repeats': 1, 'units': 256, 'act': ReLU},
        {'name': 'drop1', 'type': 'drop', 'ratio': 0.50},
        {'name': 'fc2', 'type': 'dense', 'repeats': 1, 'units': 128, 'act': ReLU},
        {'name': 'drop2', 'type': 'drop', 'ratio': 0.50}
    ]
