from typing import Union, Optional, Dict, Tuple, List

from ..base_data_loader.base_data_loader_config import BaseDataLoaderConfig


class PseudoLabelDataLoaderConfig(BaseDataLoaderConfig):
    """
        Configuration class for the Pseudo Label data loader.

        params:
            num_classes (int): Number of classes (unique labels)
                to be used for one-hot encoding the labels.
            batch_size (int) - Batch size to be used for training and evaluation.
            shuffle_buffer_size (int): Buffer size for the dataset shuffle operator.
    """

    num_classes: int = 10
    batch_size: int = 32
    shuffle_buffer_size: int = 1000
