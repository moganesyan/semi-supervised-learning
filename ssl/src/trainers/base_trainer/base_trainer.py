from abc import abstractmethod, ABC
import logging
from pickletools import optimize
from typing import Any, Dict, List, Optional

import tensorflow as tf

from .base_trainer_config import BaseTrainerConfig
from ..trainer_utils import set_seed
from ..trainer_exceptions import MissingOptimizerConfigError

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
            val_dataset (tf.data.Dataset): Validation dataset.
            scheduler (tf.keras.optimizers.schedules.LearningRateSchedule): Training parameter scheduler to be used during training.
            callbacks (List[tf.keras.callbacks.Callback]): List of callbacks to be used during training.
    """

    def __init__(
        self,
        model,
        train_dataset: tf.data.Dataset,
        training_config: BaseTrainerConfig,
        val_dataset: Optional[tf.data.Dataset] = None,
        scheduler: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> None:

        self._model = model
        self._train_dataset = train_dataset
        self._training_config = training_config
        self._val_dataset = val_dataset
        
        self._optimizer = self._get_optimizer()
        self._scheduler = scheduler
        self._callbacks = callbacks

        # set training random seed
        set_seed(training_config.seed)

    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
            Get optimizer from trainer config.

            Supported optimizers are: ['adam', 'sgd', 'rmsprop', 'custom']

            Custom optimizers (TODO) need to be implemented in the style of a keras
            optimizer, by using the `tf.keras.optimizers.Optimizer` base class.

            args:
                None
            returns:
                optimizer (tf.keras.optimizers.Optimizer) - Optimizer to be used during training.
        """

        optimizer_config = self._training_config.optimizer

        if optimizer_config is None:
            raise MissingOptimizerConfigError("The training config is missing optimizer configuration parameters.")

        if optimizer_config["name"] == "adam":
            return tf.keras.optimizers.Adam(**optimizer_config["params"])
        elif optimizer_config["name"] == "rmsprop":
            return tf.keras.optimizers.RMSprop(**optimizer_config["params"])
        elif optimizer_config["name"] == "sgd":
            return tf.keras.optimizers.SGD(**optimizer_config["params"])
        else:
            raise NotImplementedError("Custom optimizers not yet implemented.")

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
