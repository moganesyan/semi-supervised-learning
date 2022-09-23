import logging
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from ..base_trainer.base_trainer import BaseTrainer
from .pi_model_trainer_config import PiModelTrainerConfig

from ...losses.classification import categorical_cross_entropy
from ...losses.regression import pi_model_se

# set up logger
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class PiModelTrainer(BaseTrainer):
    """
        Pi-Model trainer.
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
    
    def train_step(self,
                   x_batch_labelled_1: tf.Tensor,
                   x_batch_labelled_2: tf.Tensor,
                   x_batch_unlabelled_1: tf.Tensor,
                   x_batch_unlabelled_2: tf.Tensor,
                   y_batch: tf.Tensor) -> tf.Tensor:
        """
            Apply a single training step on the input batch.

            1) Forward pass across all realisations of augmented features.
            2) Calculate loss as a weighted sum of CCE and squared error loss.
            3) Apply optimizer on the model parameters (eg: weight update).

            args:
                x_batch_labelled_1 (tf.Tensor) - First batch of augmented labelled input feature tensors.
                x_batch_labelled_2 (tf.Tensor) - Second batch of augmented labelled input feature tensors.
                x_batch_unlabelled_1 (tf.Tensor) - First batch of augmented unlabelled input feature tensors.
                x_batch_unlabelled_2 (tf.Tensor) - Second batch of augmented unlabelled input feature tensors.
                y_batch (tf.Tensor) - Batch of input label tensors.
            returns:
                total_loss (tf.Tensor) - Loss for the batch.
        """

        with tf.GradientTape() as tape:
            # forward pass calls
            y_pred_batch_labelled_1 = self._model(x_batch_labelled_1)
            y_pred_batch_labelled_2 = self._model(x_batch_labelled_2)
            y_pred_batch_unlabelled_1 = self._model(x_batch_unlabelled_1)
            y_pred_batch_unlabelled_2 = self._model(x_batch_unlabelled_2)

            # get CE loss for first batch of labelled samples only
            loss_ce = categorical_cross_entropy(y_pred_batch_labelled_1, y_batch)

            # get squared error for all
            loss_se_labelled = pi_model_se(y_pred_batch_labelled_1, y_pred_batch_labelled_2)
            loss_se_unlabelled = pi_model_se(y_pred_batch_unlabelled_1, y_pred_batch_unlabelled_2)

            # calculate weighted total loss (TODO: add weighting)
            total_loss = loss_ce + (loss_se_labelled + loss_se_unlabelled)

        self._optimizer.minimize(total_loss, self._model.trainable_variables, tape = tape)

        return total_loss


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

        for epoch in tf.range(self._training_config.num_epochs):
            train_loss = 0

            for train_step_idx, (x_batch_labelled_1_train, x_batch_labelled_2_train, x_batch_unlabelled_1_train, x_batch_unlabelled_2_train, y_batch_train) in enumerate(self._train_dataset):
                loss_train = self.train_step(
                    x_batch_labelled_1_train,
                    x_batch_labelled_2_train,
                    x_batch_unlabelled_1_train,
                    x_batch_unlabelled_2_train,
                    y_batch_train
                )
                train_loss += loss_train
            
            train_loss /=  len(self._train_dataset)

            if self._val_dataset is not None:
                val_loss = 0

                for val_step_idx, (x_batch_val, y_batch_val) in enumerate(self._val_dataset):
                    loss_val = self.eval_step(x_batch_val, y_batch_val)
                    val_loss += loss_val
                
                val_loss /= len(self._val_dataset)
                
                tf.print(f"Training loss at epoch {epoch} is : {train_loss:.2f}. Validation loss is : {val_loss:.2f}.")
            else:
                tf.print(f"Training loss at epoch {epoch} is : {train_loss:.2f}.")
