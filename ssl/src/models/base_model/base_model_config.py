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
            seed (int) - Random seed.
    """

    input_shape: Tuple[int] = None
    output_shape: Union[int, Tuple[int]] = None
    architecture: List[Dict] = None
    seed: int = 42
