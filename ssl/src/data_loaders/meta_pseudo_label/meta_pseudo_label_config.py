from typing import Union, Optional, Dict, Tuple, List

from ..base_data_loader.base_data_loader_config import BaseDataLoaderConfig


class MetaPseudoLabelDataLoaderConfig(BaseDataLoaderConfig):
    """
        Configuration class for the Meta Pseudo Label data loader.

        params:
            num_classes (int): Number of classes (unique labels)
                to be used for one-hot encoding the labels.
            batch_size (int) - Batch size to be used for training and evaluation.
            shuffle_buffer_size (int): Buffer size for the dataset shuffle operator.
            blur_params (Dict): Parameters for the gaussian blur augmenter.
            crop_params (Dict): Parameters for the random crop-and-resize augmenter.
            jitter_params (Dict): Parameters for the colour jitter augmenter.
    """

    num_classes: int = None
    batch_size: int = None
    shuffle_buffer_size: int = None

    # augmentation parameters
    blur_params: Dict = None
    crop_params: Dict = None
    jitter_params: Dict = None
