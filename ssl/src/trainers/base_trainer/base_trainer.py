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
            train_dataset (tf.data.Dataset): Training dataset.
            training_config (BaseTrainerConfig): Training config.
            val_dataset (tf.data.Dataset): Validation dataset.
    """

    def __init__(
        self,
        train_dataset: tf.data.Dataset,
        training_config: BaseTrainerConfig,
        val_dataset: Optional[tf.data.Dataset] = None) -> None:

        self._train_dataset = train_dataset
        self._training_config = training_config
        self._val_dataset = val_dataset
        
        self._lr_schedule = self._get_lr_schedule()
        self._optimizer = self._get_optimizer()
        self._callbacks = self._get_callbacks()

        # set training random seed
        set_seed(training_config.seed)

    def _get_lr_schedule(self) -> Optional[tf.keras.optimizers.schedules.LearningRateSchedule]:
        """
            Get learning rate schedule from trainer config.

            Supported schedules are: ['cosine', 'exponential', 'inverse_time']

            Custom schedules need to be implemented in the style of a keras
            learning rate schedule, by using the `tf.keras.optimizers.schedules.LearningRateSchedule`
                base class.
        
            args:
                None
            returns:
                schedule (tf.keras.optimizers.schedules.LearningRateSchedule) -
                    Learning rate schedule to be used during training.
        """

        schedule_config = self._training_config.lr_schedule

        if schedule_config is None:
            return

        if schedule_config["name"] == "cosine":
            return tf.keras.optimizers.schedules.CosineDecay(**schedule_config["params"])
        elif schedule_config["name"] == "exponential":
            return tf.keras.optimizers.schedules.ExponentialDecay(**schedule_config["params"])
        elif schedule_config["name"] == "inverse_time":
            return tf.keras.optimizers.schedules.InverseTimeDecay(**schedule_config["params"])
        else:
            raise NotImplementedError("Custom learning rate schedules not yet implemented.")  

    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
            Get optimizer from trainer config.

            Supported optimizers are: ['adam', 'sgd', 'rmsprop']

            Custom optimizers need to be implemented in the style of a keras
            optimizer, by using the `tf.keras.optimizers.Optimizer` base class.

            Will use the learning rate schedule when available.

            args:
                None
            returns:
                optimizer (tf.keras.optimizers.Optimizer) - Optimizer to be used during training.
        """

        optimizer_config = self._training_config.optimizer
        lr = optimizer_config["learning_rate"] if self._lr_schedule is None else self._lr_schedule

        if optimizer_config is None:
            raise MissingOptimizerConfigError("The training config is missing optimizer configuration parameters.")

        if lr is None:
            raise MissingOptimizerConfigError("Neither fixed learning rate nor schedule are available in the config.")

        if optimizer_config["name"] == "adam":
            return tf.keras.optimizers.Adam(learning_rate = lr, **optimizer_config["params"])
        elif optimizer_config["name"] == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate = lr, **optimizer_config["params"])
        elif optimizer_config["name"] == "sgd":
            return tf.keras.optimizers.SGD(learning_rate = lr, **optimizer_config["params"])
        else:
            raise NotImplementedError("Custom optimizers not yet implemented.")

    def _get_callbacks(self) -> tf.keras.callbacks.CallbackList:
        """
            Get callbacks from the trainer config.

            Supported callbacks are: []

            NOTE that keras pre-built callbacks won't work as they're too
                reliant on the way keras training loops are designed.

            Custom callbacks need to be implemented in the style of a keras
            callback, by using the `tf.keras.callbacks.Callback` base class.
        
            args:
                None
            returns:
                callbacks (tf.keras.callbacks.CallbackList) - Callbacks to be used during training.
        """

        callback_config = self._training_config.callbacks

        if callback_config is None:
            return

        callbacks = []

        for callback in callback_config:
            raise NotImplementedError("Custom callbacks not yet implemented.")

        callbacks = tf.keras.callbacks.CallbackList(callbacks)
        return callbacks
 
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
