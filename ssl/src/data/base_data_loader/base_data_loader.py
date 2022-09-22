from abc import ABC, abstractmethod

from typing import Union, Optional, Dict

import tensorflow as tf

from .base_data_loader_config import BaseDataLoaderConfig


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

    @abstractmethod
    def _build_data_loader(self) -> tf.data.Dataset:
        """
            Build data loader from the configuration object.

            args:
                None
            returns:
                dataset (tf.data.Dataset) - Dataset iterable.
        """

        pass

    def __call__(self) -> tf.data.Dataset:
        """
            Get data loader.

            args:
                None
            returns:
                dataset (tf.data.Dataset) - Dataset iterable.
        """

        return self._build_data_loader()
