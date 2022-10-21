from typing import Union, Optional, Dict, Tuple, List

from ..base_data_loader.base_data_loader_config import BaseDataLoaderConfig


class PiModelDataLoaderConfig(BaseDataLoaderConfig):
    """
        Configuration class for the Pi-Model data loader.

        params:
            num_classes (int): Number of classes (unique labels)
                to be used for one-hot encoding the labels.
            batch_size (int) - Batch size to be used for training and evaluation.
            batch_ratios (List[float, float]): Ratio to balance
                labelled and unlabelled samples in a batch.
            shuffle_buffer_size (int): Buffer size for the dataset shuffle operator.
            blur_params (Dict): Parameters for the gaussian blur augmenter.
            crop_params (Dict): Parameters for the random crop-and-resize augmenter.
            jitter_params (Dict): Parameters for the colour jitter augmenter.
    """

    num_classes: int = 10
    batch_size: int = 32
    batch_ratios: List[float] = [0.50, 0.50]
    shuffle_buffer_size: int = 1000

    # augmentation parameters
    blur_params: Dict = {
        'chance': 0.33,
        'kernel_ratio': 0.10,
        'blur_strength': (0.1, 2.0)
    }

    crop_params: Dict = {
        'chance': 0.33,
        'crop_size': (0.08, 1.0),
        'aspect_range': (0.75, 1.33),
        'num_tries': 100
    }

    jitter_params: Dict = {
        'chance': 0.33,
        'distort_strength': 0.50
    }
