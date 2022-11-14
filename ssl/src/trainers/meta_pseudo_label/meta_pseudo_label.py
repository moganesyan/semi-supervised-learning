import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from .meta_pseudo_label_config import MetaPseudoLabelTrainerConfig
from ..base_trainer.base_trainer import BaseTrainer
from ..trainer_exceptions import MissingOptimizerConfigError

from ...models.base_model.base_model import BaseModel

from ...losses.classification import categorical_cross_entropy_masked
from ...losses.classification import categorical_cross_entropy

# set up logger
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class MetaPseudoLabelTrainer(BaseTrainer):
    """
        Meta Pseudo Label trainer.
        As seen in the original paper: https://arxiv.org/abs/2003.10580
    """

    def __init__(
        self,
        student_model: BaseModel,
        teacher_model: BaseModel,
        train_dataset: tf.data.Dataset,
        training_config: MetaPseudoLabelTrainerConfig,
        val_dataset: Optional[tf.data.Dataset] = None) -> None:

        super().__init__(
            train_dataset,
            training_config,
            val_dataset
        )
        self._student_model = student_model
        self._teacher_model = teacher_model

        self._student_optimizer, self._teacher_optimizer = self._get_twin_optimizers()

    def _get_twin_optimizers(self) -> Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.Optimizer]:
        """
            Get student and teacher optimizers from trainer config.

            Supported optimizers are: ['adam', 'sgd', 'rmsprop']

            Custom optimizers need to be implemented in the style of a keras
            optimizer, by using the `tf.keras.optimizers.Optimizer` base class.

            Will use the learning rate schedule when available.

            args:
                None
            returns:
                student_optimizer (tf.keras.optimizers.Optimizer) - Student optimizer to be used during training.
                teacher_optimizer (tf.keras.optimizers.Optimizer) - Teacher optimizer to be used during training.
        """

        optimizer_config = self._training_config.optimizer
        lr = optimizer_config["learning_rate"] if self._lr_schedule is None else self._lr_schedule

        if optimizer_config is None:
            raise MissingOptimizerConfigError("The training config is missing optimizer configuration parameters.")

        if lr is None:
            raise MissingOptimizerConfigError("Neither fixed learning rate nor schedule are available in the config.")

        if optimizer_config["name"] == "adam":
            student_optimizer = tf.keras.optimizers.Adam(learning_rate = lr, **optimizer_config["params"])
            teacher_optimizer = tf.keras.optimizers.Adam(learning_rate = lr, **optimizer_config["params"])
            return student_optimizer, teacher_optimizer
        elif optimizer_config["name"] == "rmsprop":
            student_optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr, **optimizer_config["params"])
            teacher_optimizer = tf.keras.optimizers.RMSprop(learning_rate = lr, **optimizer_config["params"])
            return student_optimizer, teacher_optimizer
        elif optimizer_config["name"] == "sgd":
            student_optimizer = tf.keras.optimizers.SGD(learning_rate = lr, **optimizer_config["params"])
            teacher_optimizer = tf.keras.optimizers.SGD(learning_rate = lr, **optimizer_config["params"])
            return student_optimizer, teacher_optimizer
        else:
            raise NotImplementedError("Custom optimizers not yet implemented.")

    def train_step(self,
                   x_batch: tf.Tensor,
                   y_batch: tf.Tensor,
                   mask_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
            Apply a single training step on the input batch.

        1) Inferece pass on unlabelled data using the teacher model
            to get (hard) pseudo labels. (unwatched)
        2) Compute the student model's CE loss on unsupervised
            samples against pseudo labels.
        3) Update student model's weights using SGD.
        4) Compute teacher model's feedback coefficient using updated student model's
            CE loss on labelled data and old student model's CE loss on unlabelled data
            against pseudo labels (reused from step 3).
        5) Compute teacher model's feedback gradient using the feedback coefficient from
            step 4 and teacher model's CE loss on unlabelled data using its own hard pseudo labels.
        6) Compute the teacher model's supervised gradient by using the CE loss on labelled samples.
        6a) TODO: Compute the teacher model's unsupervised UDA gradient.
        7) Update teacher model's weights using the gradients calculated in steps 5 and 6 using SGD.

            args:
                x_batch (tf.Tensor) - Batch of input feature tensors containing labelled and
                    and unlabelled samples.
                y_batch (tf.Tensor) - Batch of input label tensors.
                mask_batch (tf.Tensor) - Batch of flags indicating the labelled status of each sample.
            returns:
                loss_student_unsup (tf.Tensor) - Unsupervised loss for the student model against pseudo labels.
                loss_student_sup (tf.Tensor) - Supervised loss for the student model against real labels.
                loss_teacher_unsup (tf.Tensor) - Unsupervised loss for the teacher model against pseudo labels.
                loss_teacher_sup (tf.Tensor) - Supervised loss for the teacher model against real labels.
        """

        # Inference on unlabelled data using the teacher model to get pseudo labels.
        num_classes = tf.shape(y_batch)[1]
        y_batch_pseudo_soft = self._teacher_model(x_batch, training = False)
        y_batch_pseudo = tf.argmax(y_batch_pseudo_soft, axis = 1)
        y_batch_pseudo = tf.one_hot(y_batch_pseudo, num_classes)
        
        # Compute student model's loss on unlaballed data.
        with tf.GradientTape() as tape_student_old:
            # Student model forward pass
            y_batch_student_pred = self._student_model(x_batch)
            
            # Unsupervised student loss
            loss_student_unsup = categorical_cross_entropy_masked(
                y_batch_student_pred,
                y_batch_pseudo,
                tf.logical_not(mask_batch)
            )

        # Update student model's weights
        student_grads_old = tape_student_old.gradient(
            loss_student_unsup,
            self._student_model.trainable_variables
        )
        self._student_optimizer.apply_gradients(
            zip(student_grads_old,
            self._student_model.trainable_variables)
        )
        
        # Compute updated student model's loss on labelled data.
        with tf.GradientTape() as tape_student_new:
            y_batch_student_pred_new = self._student_model(x_batch)
            loss_student_sup = categorical_cross_entropy_masked(
                y_batch_student_pred_new,
                y_batch,
                mask_batch
            )
        
        # Compute the teacher model's feedback coefficient
        student_grads_new = tape_student_new.gradient(
            loss_student_sup,
            self._student_model.trainable_variables
        )
        dot_product_temp = tf.reduce_sum(
            tf.multiply(
                tf.concat([tf.reshape(g, (-1,1)) for g in student_grads_new], 0),
                tf.concat([tf.reshape(g, (-1,1)) for g in student_grads_old], 0)
            )
        )
        h = self._student_optimizer.learning_rate * dot_product_temp
        
        # Compute losses for the teacher model
        with tf.GradientTape(persistent = True) as tape_teacher:
            # Forward call using the teacher model
            y_batch_teacher_pred = self._teacher_model(x_batch)
            
            # Get teacher model's unsupervised loss (against its own hard pseudo labels)
            loss_teacher_unsup = categorical_cross_entropy_masked(
                y_batch_teacher_pred,
                y_batch_pseudo,
                tf.logical_not(mask_batch)
            )
            
            # Get teacher model's supervised loss
            loss_teacher_sup = categorical_cross_entropy_masked(
                y_batch_teacher_pred,
                y_batch,
                mask_batch
            )
        
        # Compute gradients for the teacher model
        teacher_grads_unsup = h * tape_teacher.gradient(
            loss_teacher_unsup,
            self._teacher_model.trainable_variables
        )
        
        teacher_grads_sup = tape_teacher.gradient(
            loss_teacher_sup,
            self._teacher_model.trainable_variables
        )
        
        # TODO: Unsupervised UDA loss for teacher
        
        # Compute total gradient for the teacher model
        teacher_grads_total = [x + y for (x,y) in zip(teacher_grads_unsup, teacher_grads_sup)]
        
        # Update teacher model's weights
        self._teacher_optimizer.apply_gradients(
            zip(teacher_grads_total,
            self._teacher_model.trainable_variables)
        )

        return loss_student_unsup, loss_student_sup, loss_teacher_unsup, loss_teacher_sup

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

        for epoch in tf.range(self._training_config.num_epochs, dtype = tf.int64):

            # instantiate training losses
            ltsu = tf.constant(0, tf.float32)
            ltss = tf.constant(0, tf.float32)
            lttu = tf.constant(0, tf.float32)
            ltts = tf.constant(0, tf.float32)

            for train_step_idx, (x_batch_train, y_batch_train, mask_batch_train) in enumerate(self._train_dataset):
                _ltsu, _ltss, _lttu, _ltts  = self.train_step(
                    x_batch_train,
                    y_batch_train,
                    mask_batch_train,
                )

                ltsu += _ltsu
                ltss += _ltss
                lttu += _lttu
                ltts += _ltts
            
            ltsu /=  train_step_idx
            ltss /=  train_step_idx
            lttu /=  train_step_idx
            ltts /=  train_step_idx

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
                    "Training loss at epoch {epoch} is : ltsu {ltsu:.2f}, ltss {ltss:.2f}, lttu {lttu:.2f}, ltts {ltts:.2f}. Validation loss is : {val_loss:.2f}. Validation acc. is : {val_acc:.2f}."
                )

            else:
                tf.print(f"Training loss at epoch {epoch} is : ltsu {ltsu:.2f}, ltss {ltss:.2f}, lttu {lttu:.2f}, ltts {ltts:.2f}.")
