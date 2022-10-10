from typing import Union, Optional, Dict, Tuple, List

from ..base_data_loader.base_data_loader_config import BaseDataLoaderConfig


class CategoricalCEDataLoaderConfig(BaseDataLoaderConfig):
    """
        Configuration class for the categorical CE data loader.

        params:
            num_classes (int): Number of classes (unique labels)
                to be used for one-hot encoding the labels.
            batch_size (int) - Batch size to be used for training and evaluation.
            shuffle_buffer_size (int): Buffer size for the dataset shuffle operator.
            blur_chance (float): Probability (decimal) at which random blur augmentor should be applied.
            crop_chance (float): Probability (decimal) at which random crop & resize should be applied.
            jitter_chance (float): Probability (decimal) at which random colour jitter should be applied.
    """

    num_classes: int = 10
    batch_size: int = 32
    shuffle_buffer_size: int = 1000
    blur_chance: float = 0.05
    crop_chance: float = 0.10
    jitter_chance: float = 0.20
