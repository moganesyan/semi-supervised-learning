import logging
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from .mean_teacher_config import MeanTeacherTrainerConfig
from ..base_trainer.base_trainer import BaseTrainer

from ...models.base_model.base_model import BaseModel

from ...losses.classification import categorical_cross_entropy_masked
from ...losses.classification import categorical_cross_entropy
from ...losses.consistency import pi_model_se

# set up logger
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class MeanTeacherTrainer(BaseTrainer):
    """
        Mean Teacher trainer.
        As seen in the original paper: https://arxiv.org/abs/1703.01780
    """

    def __init__(
        self,
        student_model: BaseModel,
        teacher_model: BaseModel,
        train_dataset: tf.data.Dataset,
        training_config: MeanTeacherTrainerConfig,
        val_dataset: Optional[tf.data.Dataset] = None) -> None:

        super().__init__(
            train_dataset,
            training_config,
            val_dataset
        )
        self._student_model = student_model
        self._teacher_model = teacher_model

    def _update_teacher_weights(self):
        """
            Update the teacher model weights as the exponential
                moving average (EMA) of student model weights.
            
            args:
                None
            returns:
                None
        """

        student_weights = self._student_model.get_weights()
        teacher_weights = self._teacher_model.get_weights()

        teacher_weights_new = []

        for (student_weight, teacher_weight) in zip(student_weights, teacher_weights):
            new_teacher_weight = ((self._training_config.alpha * teacher_weight)
                + ((1 - self._training_config.alpha) * student_weight))
            teacher_weights_new.append(new_teacher_weight)
        
        self._teacher_model.set_weights(teacher_weights_new)

    def train_step(self,
                   x_batch_1: tf.Tensor,
                   x_batch_2: tf.Tensor,
                   y_batch: tf.Tensor,
                   mask_batch: tf.Tensor,
                   loss_weight: tf.Tensor) -> tf.Tensor:
        """
            Apply a single training step on the input batch.

            1) Forward pass on all features (labelled and unlabelled).
                Student model assigned the first realisation of augmented features.
                Teacher model assigned the second realisation of augmented features.
            2) Calculate masked CCE for labelled samples. only and
                calculate consistency loss for all samples.
            3) Calculate total loss as a weighted sum of labelled and unlabelled components.
            4) Apply optimizer on the model parameters (eg: weight update).
            5) Update teacher weights using exponential moving average (EMA)
                of student's weights.

            args:
                x_batch_1 (tf.Tensor) - Batch of input feature tensors containing labelled and
                    and unlabelled samples. (1st realistation of augmentations)
                x_batch_2 (tf.Tensor) - Batch of input feature tensors containing labelled and
                    and unlabelled samples. (2nd realistation of augmentations)
                y_batch (tf.Tensor) - Batch of input label tensors.
                mask_batch (tf.Tensor) - Batch of flags indicating the labelled status of each sample.
                loss_weight (tf.Tensor) - Weight coefficient to balance supervised and unsupervised loss
                    components.
            returns:
                total_loss (tf.Tensor) - Loss for the batch.
        """

        with tf.GradientTape() as tape:
            # forward pass calls
            y_pred_batch_1 = self._student_model(x_batch_1)

            with tape.stop_recording():
                y_pred_batch_2 = self._teacher_model(x_batch_2)

            # compute CE loss
            loss_ce = categorical_cross_entropy_masked(
                y_pred_batch_1,
                y_batch,
                mask_batch
            )

            # get squared error for all
            loss_se = pi_model_se(
                y_pred_batch_1,
                y_pred_batch_2
            )

            total_loss = loss_ce + (loss_weight * loss_se)

        # iters = self._optimizer.iterations
        # if iters % 250 == 0:
        #     if not tf.math.is_nan(total_loss):
        #         tf.print(tf.strings.format("CE loss at iteration {}: {}", (iters, loss_ce)))
        #         tf.print(tf.strings.format("SE loss at iteration {}: {}", (iters, loss_se)))
        #         tf.print(tf.strings.format("Mask sum at iteration {}: {}", (iters, tf.reduce_sum(tf.cast(mask_batch, tf.float32)))))
        #     else:
        #         tf.print(tf.strings.format("NaN loss at iteration: {}", (iters)))

        self._optimizer.minimize(
            total_loss,
            self._student_model.trainable_variables,
            tape = tape
        )

        # update teacher model weights
        self._update_teacher_weights()

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

        y_pred_batch = self._student_model(x_batch, training = False)
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

        for epoch in tf.range(self._training_config.num_epochs, dtype = tf.int64):
            # pick loss weight
            if epoch < self._training_config.loss_ramp_up_epochs:
                loss_weight = loss_weights[epoch]
            else:
                loss_weight = tf.constant(self._training_config.unsup_loss_weight, tf.float32)

            train_loss = tf.constant(0, tf.float32)
            for train_step_idx, (x_batch_1_train, x_batch_2_train, y_batch_train, mask_batch_train) in enumerate(self._train_dataset):
                loss_train = self.train_step(
                    x_batch_1_train,
                    x_batch_2_train,
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
