from typing import Union, Tuple, List, Dict

from ..base_model.base_model_config import BaseModelConfig

from tensorflow.keras.layers import ReLU, Softmax, Layer


class Lenet5Config(BaseModelConfig):
    """
        Lenet5 configuration class.

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
        {'name': 'conv1', 'type': 'conv', 'num_blocks': 1, 'filters': 6, 'k': 5, 's': 2, 'pad': 'valid', 'act':  ReLU},
        {'name': 'conv2', 'type': 'conv', 'num_blocks': 1, 'filters': 16, 'k': 5, 's': 2, 'pad': 'valid', 'act': ReLU},
        {'name': 'conv3', 'type': 'conv', 'num_blocks': 1, 'filters': 120, 'k': 5, 's': 1, 'pad': 'valid', 'act': ReLU},
        {'name': 'flat1', 'type': 'flat', 'num_blocks': 1},
        {'name': 'fc1', 'type': 'dense', 'num_blocks': 1, 'units': 84, 'act': ReLU}
    ]