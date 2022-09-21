import logging
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from ..base_trainer.base_trainer import BaseTrainer
from .pi_model_trainer_config import PiModelTrainerConfig

from ...losses.classification import categorical_cross_entropy

# set up logger
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class PiModelTrainer(BaseTrainer):
    """
        Pi-model trainer.
        As seen in the original paper: https://arxiv.org/abs/1610.02242
    """

    def __init__(
        self,
        model,
        train_dataset: tf.data.Dataset,
        training_config: PiModelTrainerConfig,
        val_dataset: Optional[tf.data.Dataset] = None,
        scheduler: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> None:
        super().__init__(
            model, train_dataset, training_config,
            val_dataset, scheduler, callbacks
        )
    
    def train_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        """
            Apply a single training step on the input batch.

            1) Forward pass.
            2) Calculate CCE loss.
            3) Apply optimizer on the model parameters (eg: weight update).

            args:
                x_batch (tf.Tensor) - Batch of input feature tensors.
                y_batch (tf.Tensor) - Batch of input label tensors.
            returns:
                loss (tf.Tensor) - Loss for the batch.
        """

        raise NotImplementedError


    def eval_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> float:
        """
            Apply a single evaluation step on the input batch.

            1) Forward pass.
            2) Calculate CCE loss.

            args:
                x_batch (tf.Tensor) - Batch of input feature tensors.
                y_batch (tf.Tensor) - Batch of input label tensors.
            returns:
                loss (tf.Tensor) - Loss for the batch.
        """

        raise NotImplementedError
    
    def train(self) -> None:
        """
            Training loop.

            1) Repeat for`num_epochs`
            2) For each epoch, loop through the training dataset batchwise.
                For each batch, apply the training step.
            3) For each epoch, loop through the validation dataset (if available) batchwise.
                For each batch, apply the evaluation step.
            4) Print training and evaluation metrics.

            args:
                None
            returns:
                None
        """

        raise NotImplementedError
