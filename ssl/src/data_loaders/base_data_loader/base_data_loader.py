from abc import ABC, abstractmethod

from typing import Union, Optional, Dict

import tensorflow as tf

from .base_data_loader_config import BaseDataLoaderConfig
from ..data_loader_utils import set_seed


class BaseDataLoader(ABC):
    """
        Base class for all data loaders.

        All data loader classes should be derived from here.

        args:
            data_loader_config (BaseDataLoaderConfig): Configuration class for the data loader.
        returns:
            None
    """

    def __init__(self, data_loader_config: BaseDataLoaderConfig) -> None:
        self._data_loader_config = data_loader_config
        set_seed(data_loader_config.seed)

    @abstractmethod
    def _build_data_loader(self, training: bool) -> tf.data.Dataset:
        """
            Build data loader from the configuration object.

            args:
                training: bool - Flag to indicate if the dataset is for training.
            returns:
                dataset (tf.data.Dataset) - Dataset iterable.
        """

        pass

    def __call__(self, training: bool = True) -> tf.data.Dataset:
        """
            Get data loader.

            args:
                training: bool - Flag to indicate if the dataset is for training.
            returns:
                dataset (tf.data.Dataset) - Dataset iterable.
        """

        return self._build_data_loader(training)
