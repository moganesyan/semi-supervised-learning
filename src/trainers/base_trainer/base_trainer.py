from abc import abstractmethod, ABC
import logging
from typing import Any, Dict, List, Optional

import tensorflow as tf

from .base_trainer_config import BaseTrainerConfig

# set up logger
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class BaseTrainer(ABC):
    """
        Base class for model training.

        args:
            model (BaseModel): Model to train.
            train_dataset (tf.data.Dataset): Training dataset.
            training_config (BaseTrainerConfig): Training config.
            optimizer (tf.keras.optimizers.Optimizer): Optimizer to be used during training.
            val_dataset (tf.data.Dataset): Validation dataset.
            scheduler (tf.keras.optimizers.schedules.LearningRateSchedule): Training parameter scheduler to be used during training.
            callbacks (List[tf.keras.callbacks.Callback]): List of callbacks to be used during training.
    """

    def __init__(
        self,
        model,
        train_dataset: tf.data.Dataset,
        training_config: BaseTrainerConfig,
        optimizer: tf.keras.optimizers.Optimzer,
        val_dataset: Optional[tf.data.Dataset] = None,
        scheduler: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> None:

        self._model = model
        self._train_dataset = train_dataset
        self._training_config = training_config
        self._optimizer = optimizer
        self._val_dataset = val_dataset
        self._scheduler = scheduler
        self._callbacks = callbacks

    @abstractmethod
    def train_step(self) -> None:
        """
            Single training step.
        """

        pass

    @abstractmethod
    def eval_step(self) -> None:
        """
            Single evaludation step.
        """

        pass

    @abstractmethod
    def train(self) -> None:
        """
            Training loop.
        """

        pass

    def save_model(self) -> None:
        """
            Save model.
        """

        pass

    def save_checkpoint(self) -> None:
        """
            Save model checkpoint.
        """

        pass
