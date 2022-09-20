from platform import architecture
from typing import Dict, List, Union, Tuple


class BaseModelConfig():
    """
        Base configuration class for all model architectures.

        All architecture configuration classes to be derived from here.

        params:
            input_shape (Tuple[int]) - Input tensor shape.
            output_shape (Union[int, Tuple[int]]) - Output tensor shape.
            architecture (Dict) - Architecture config object.
    """

    input_shape: Tuple[int] = (224, 224, 3)
    output_shape: Union[int, Tuple[int]] = 10
    architecture: List[Dict] = None
