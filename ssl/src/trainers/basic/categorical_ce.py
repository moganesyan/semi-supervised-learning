import logging
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from ..base_trainer.base_trainer import BaseTrainer
from .categorical_ce_config import CategoricalCETrainerConfig

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
        training_config: CategoricalCETrainerConfig,
        val_dataset: Optional[tf.data.Dataset] = None) -> None:
        super().__init__(
            model, train_dataset,
            training_config, val_dataset
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

        with tf.GradientTape() as tape:
            y_pred_batch = self._model(x_batch)
            loss = categorical_cross_entropy(y_pred_batch, y_batch)
        self._optimizer.minimize(loss, self._model.trainable_variables, tape = tape)

        return loss

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

        y_pred_batch = self._model(x_batch, training = False)
        loss = categorical_cross_entropy(y_pred_batch, y_batch)
        return loss
    
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

        if self._callbacks is not None:
            self._callbacks.on_train_begin()

        for epoch in tf.range(self._training_config.num_epochs):
            if self._callbacks is not None:
                self._callbacks.on_epoch_begin(epoch)

            train_loss = tf.constant(0, dtype = tf.float32)
            train_step_idx = tf.constant(0, dtype = tf.float32)

            for x_batch_train, y_batch_train in self._train_dataset:
                if self._callbacks is not None:
                    self._callbacks.on_train_batch_begin(train_step_idx)

                loss_train = self.train_step(x_batch_train, y_batch_train)
                train_loss += loss_train

                if self._callbacks is not None:
                    self._callbacks.on_train_batch_end(
                    train_step_idx,
                    logs = {'loss': loss_train}
                    )

                train_step_idx += 1
            
            train_loss /=  train_step_idx

            if self._val_dataset is not None:
                val_loss = tf.constant(0, dtype = tf.float32)
                val_step_idx = tf.constant(0, dtype = tf.float32)

                for x_batch_val, y_batch_val in self._val_dataset:
                    loss_val = self.eval_step(x_batch_val, y_batch_val)
                    val_loss += loss_val
                    val_step_idx += 1
                
                val_loss /= val_step_idx
                
                tf.print(f"Training loss at epoch {epoch} is : {train_loss:.2f}. Validation loss is : {val_loss:.2f}.")

                if self._callbacks is not None:
                    self._callbacks.on_epoch_end(
                        epoch,
                        logs = {'loss': train_loss, 'val_loss': val_loss}
                    )
            else:
                tf.print(f"Training loss at epoch {epoch} is : {train_loss:.2f}.")

                if self._callbacks is not None:
                    self._callbacks.on_epoch_end(
                        epoch,
                        logs = {'loss': train_loss}
                    )

        if self._callbacks is not None:
            if self._val_dataset is not None:
                self._callbacks.on_train_end(logs = {'loss': train_loss, 'val_loss': val_loss})
            else:
                self._callbacks.on_train_end(logs = {'loss': train_loss})
