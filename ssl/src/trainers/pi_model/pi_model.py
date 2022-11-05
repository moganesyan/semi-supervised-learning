import logging
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from ..base_trainer.base_trainer import BaseTrainer
from .pi_model_config import PiModelTrainerConfig

from ...losses.classification import categorical_cross_entropy
from ...losses.consistency import pi_model_se

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
        val_dataset: Optional[tf.data.Dataset] = None) -> None:
        super().__init__(
            model, train_dataset,
            training_config, val_dataset
        )

    def train_step(self,
                   x_batch_labelled_1: tf.Tensor,
                   x_batch_labelled_2: tf.Tensor,
                   x_batch_unlabelled_1: tf.Tensor,
                   x_batch_unlabelled_2: tf.Tensor,
                   y_batch: tf.Tensor,
                   loss_weight: tf.Tensor) -> tf.Tensor:
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
                loss_weight (tf.Tensor) - Weight coefficient to balance supervised and unsupervised loss
                    components.
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

            # compute CE loss
            loss_ce = categorical_cross_entropy(y_pred_batch_labelled_1, y_batch)
            not_nan_ce = tf.dtypes.cast(
                tf.math.logical_not(tf.math.is_nan(loss_ce)),
                dtype=tf.float32
            )
            loss_ce = tf.math.multiply_no_nan(
                loss_ce,
                not_nan_ce
            )

            # get squared error for all
            loss_se_labelled = pi_model_se(y_pred_batch_labelled_1, y_pred_batch_labelled_2)
            not_nan_se_labelled = tf.dtypes.cast(
                tf.math.logical_not(tf.math.is_nan(loss_se_labelled)),
                dtype=tf.float32
            )
            loss_se_labelled = tf.math.multiply_no_nan(
                loss_se_labelled,
                not_nan_se_labelled
            )

            loss_se_unlabelled = pi_model_se(y_pred_batch_unlabelled_1, y_pred_batch_unlabelled_2)
            not_nan_se_unlabelled = tf.dtypes.cast(
                tf.math.logical_not(tf.math.is_nan(loss_se_unlabelled)),
                dtype=tf.float32
            )
            loss_se_unlabelled = tf.math.multiply_no_nan(
                loss_se_unlabelled,
                not_nan_se_unlabelled
            )

            # total loss
            loss_se = loss_se_labelled + loss_se_unlabelled

            total_loss = loss_ce + (loss_weight * loss_se)

        # iters = self._optimizer.iterations
        # if iters % 250 == 0:
        #     if not tf.math.is_nan(total_loss):
        #         # tf.print(x_batch_labelled_1, y_batch)
        #         tf.print(tf.strings.format("CE loss at iteration {}: {}", (iters, loss_ce)))
        #         # tf.print(tf.strings.format("SE loss (labelled) at iteration {}: {}", (iters, loss_se_labelled)))
        #         # tf.print(tf.strings.format("SE loss (unlabelled) at iteration {}: {}", (iters, loss_se_unlabelled)))
        #         tf.print(tf.strings.format("SE loss at iteration {}: {}", (iters, loss_se)))
        #         # tf.print(tf.strings.format("output vectors (unlabelled) 1: {}", (y_pred_batch_unlabelled_1)))
        #         # tf.print(tf.strings.format("output vectors (unlabelled) 2: {}", (y_pred_batch_unlabelled_2)))
        #     else:
        #         tf.print(tf.strings.format("NaN loss at iteration: {}", (iters)))

        self._optimizer.minimize(
            total_loss,
            self._model.trainable_variables,
            tape = tape
        )

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
                matches (tf.Tensor) - Binary tensor indicating whether
                    the prediction matches the true label.
        """

        y_pred_batch = self._model(x_batch, training = False)
        loss = categorical_cross_entropy(y_pred_batch, y_batch)

        y_batch_label = tf.math.argmax(y_batch, axis = -1)
        y_batch_predicted = tf.math.argmax(y_pred_batch, axis = -1)

        matches = tf.cast(tf.equal(y_batch_label, y_batch_predicted), tf.float32)
        return loss, matches
    
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

        # loss function weighting using gaussian rampup
        loss_t = tf.linspace(
            0.0,
            1.0,
            self._training_config.loss_ramp_up_epochs
        )
        loss_weights = self._training_config.unsup_loss_weight * tf.exp(
            tf.constant(-5, tf.float32) * tf.math.pow((tf.constant(1,tf.float32) - loss_t),2)
        )

        for epoch in tf.range(self._training_config.num_epochs, dtype = tf.int32):
            # pick loss weight
            if epoch < self._training_config.loss_ramp_up_epochs:
                loss_weight = loss_weights[epoch]
            else:
                loss_weight = tf.constant(self._training_config.unsup_loss_weight, tf.float32)

            train_loss = tf.constant(0, tf.float32)
            for train_step_idx, (x_batch_labelled_1_train, x_batch_labelled_2_train, x_batch_unlabelled_1_train, x_batch_unlabelled_2_train, y_batch_train) in enumerate(self._train_dataset):
                loss_train = self.train_step(
                    x_batch_labelled_1_train,
                    x_batch_labelled_2_train,
                    x_batch_unlabelled_1_train,
                    x_batch_unlabelled_2_train,
                    y_batch_train,
                    loss_weight
                )

                train_loss += loss_train
            
            train_loss /=  train_step_idx

            if self._val_dataset is not None:

                matches_val = []
                val_loss = tf.constant(0, tf.float32)
                for val_step_idx, (x_batch_val, y_batch_val) in enumerate(self._val_dataset):
                    loss_val, match_val = self.eval_step(x_batch_val, y_batch_val)

                    matches_val.append(match_val.numpy())
                    val_loss += loss_val
                
                val_loss /= val_step_idx
                matches_val = np.concatenate(matches_val)
                val_acc = 100 * (np.sum(matches_val) / len(matches_val))
                
                tf.print(
                    f"Training loss at epoch {epoch} is : {train_loss:.2f}. Validation loss is : {val_loss:.2f}. Validation acc. is : {val_acc:.2f}."
                )

            else:
                tf.print(f"Training loss at epoch {epoch} is : {train_loss:.2f}.")
