import logging
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from .pseudo_label_config import PseudoLabelTrainerConfig
from ..base_trainer.base_trainer import BaseTrainer

from ...models.base_model.base_model import BaseModel

from ...losses.classification import categorical_cross_entropy_masked
from ...losses.classification import categorical_cross_entropy

# set up logger
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class PseudoLabelTrainer(BaseTrainer):
    """
        Pseudo Label trainer.

        As seen in the original paper: [Lee, Dong-Hyun., (2013)]

        args:
            model (BaseModel) - Keras model object.
        returns:
            None
    """

    def __init__(
        self,
        model: BaseModel,
        train_dataset: tf.data.Dataset,
        training_config: PseudoLabelTrainerConfig,
        val_dataset: Optional[tf.data.Dataset] = None) -> None:

        super().__init__(
            train_dataset,
            training_config,
            val_dataset
        )
        self._model = model

    def train_step(self,
                   x_batch: tf.Tensor,
                   y_batch: tf.Tensor,
                   mask_batch: tf.Tensor,
                   loss_weight: tf.Tensor) -> tf.Tensor:
        """
            Apply a single training step on the input batch.

            1) Forward pass on all features (labelled and unlabelled).
            2) Run inference on all features (labelled and unlabelled).
            3) Calculate masked CCE for both labelled and unlabelled mini-batches
                based on the `mask_batch` parameter.
            4) Calculate total loss as a weighted sum of labelled and unlabelled components.
            5) Apply optimizer on the model parameters (eg: weight update).

            args:
                x_batch(tf.Tensor) - Batch of input feature tensors containing labelled and
                    and unlabelled samples.
                y_batch (tf.Tensor) - Batch of input label tensors.
                mask_batch (tf.Tensor) - Batch of flags indicating the labelled status of each sample.
                loss_weight (tf.Tensor) - Weight coefficient to balance supervised and unsupervised loss
                    components.
            returns:
                total_loss (tf.Tensor) - Loss for the batch.
        """

        with tf.GradientTape() as tape:
            # forward pass calls
            y_pred_batch = self._model(x_batch)

            # create pseudo labels
            with tape.stop_recording():
                num_classes = tf.shape(y_batch)[1]
                y_batch_pseudo = self._model(x_batch, training = False)
                y_batch_pseudo = tf.argmax(y_batch_pseudo, 1)
                y_batch_pseudo_onehot = tf.one_hot(
                    y_batch_pseudo,
                    num_classes
                )

            # compute CE loss on the labelled batch
            loss_ce = categorical_cross_entropy_masked(
                y_pred_batch,
                y_batch,
                mask_batch
            )

            # get CE loss on the unlabelled batch using pseudolabels
            loss_ce_pseudo = categorical_cross_entropy_masked(
                y_pred_batch,
                y_batch_pseudo_onehot,
                tf.math.logical_not(mask_batch)
            )

            # total loss
            total_loss = loss_ce + (loss_weight * loss_ce_pseudo)

        # iters = self._optimizer.iterations
        # if iters % 250 == 0:
        #     if not tf.math.is_nan(total_loss):
        #         tf.print(tf.strings.format("CE loss at iteration {}: {}", (iters, loss_ce)))
        #         tf.print(tf.strings.format("CE (pseudo) loss at iteration {}: {}", (iters, loss_ce_pseudo)))
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
    
    def _get_loss_weight(self, epoch: int) -> float:
        """
            Get unsupervised loss component weight.

            args:
                epoch (int): Current epoch number.
            returns:
                loss_weight (float): Loss weight.
        """

        if epoch < self._training_config.t1:
            loss_weight = tf.constant(0, tf.float32)
        elif epoch >= self._training_config.t1 and epoch < self._training_config.t2:
            numer = tf.cast(epoch, tf.float32) - self._training_config.t1
            denom = self._training_config.t2 - self._training_config.t1
            loss_weight = self._training_config.alpha * (numer / denom)
        else:
            loss_weight = tf.constant(self._training_config.alpha, tf.float32)

        return loss_weight

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

        for epoch in tf.range(self._training_config.num_epochs, dtype = tf.int64):
            # get loss weight
            loss_weight = self._get_loss_weight(epoch)

            train_loss = tf.constant(0, tf.float32)
            for train_step_idx, (x_batch_train, y_batch_train, mask_batch_train) in enumerate(self._train_dataset):
                loss_train = self.train_step(
                    x_batch_train,
                    y_batch_train,
                    mask_batch_train,
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
