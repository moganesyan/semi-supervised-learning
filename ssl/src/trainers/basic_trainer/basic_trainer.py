import enum
from errno import ENEEDAUTH
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from ..base_trainer.base_trainer import BaseTrainer
from ..base_trainer.base_trainer_config import BaseTrainerConfig

from ...losses.classification import categorical_cross_entropy

# set up logger
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class CategoricalCETrainer(BaseTrainer):
    """
        Categorical Cross Entropy (CE) trainer.
    """

    def __init__(
        self,
        model,
        train_dataset: tf.data.Dataset,
        training_config: BaseTrainerConfig,
        optimizer: tf.keras.optimizers.Optimizer,
        val_dataset: Optional[tf.data.Dataset] = None,
        scheduler: Optional[tf.keras.optimizers.schedules.LearningRateSchedule] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None) -> None:
        super().__init__(
            model, train_dataset, training_config,
            optimizer, val_dataset, scheduler, callbacks
        )
    
    def train_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> float:
        """
            Training step.
        """

        with tf.GradientTape() as tape:
            y_pred_batch = self._model(x_batch)
            loss = categorical_cross_entropy(y_pred_batch, y_batch)
        self._optimizer.minimize(loss, self._model.trainable_variables, tape = tape)

        return loss

    def eval_step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> float:
        """
            Evaluation step.
        """

        y_pred_batch = self._model(x_batch)
        loss = categorical_cross_entropy(y_pred_batch, y_batch)
        return loss
    
    def train(self) -> None:
        """
            Training loop.
        """

        train_losses = []
        for epoch in range(self._training_config.num_epochs):
            for train_step_idx, (x_batch_train, y_batch_train) in enumerate(self._train_dataset):
                loss_train = self.train_step(x_batch_train, y_batch_train)
                train_losses.append(loss_train)

            val_losses = []
            if self._val_dataset is not None:
                for val_step_idx, (x_batch_val, y_batch_val) in enumerate(self._val_dataset):
                    loss_val = self.eval_step(x_batch_val, y_batch_val)
                    val_losses.append(loss_val)
                
                logger.info(f"Validation loss at epoch {epoch} is: {np.mean(val_losses):.2f}")

        return train_losses
