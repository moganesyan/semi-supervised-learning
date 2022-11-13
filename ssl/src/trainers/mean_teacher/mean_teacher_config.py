from typing import Union, Optional, Dict

from ..base_trainer.base_trainer_config import BaseTrainerConfig


class MeanTeacherTrainerConfig(BaseTrainerConfig):
    """
        Configuration class for the Mean Teacher trainer.
        As seen in the original paper: https://arxiv.org/abs/1703.01780

        params:
            output_dir (str): Directory where output models and checkpoints
                will be stored.
            num_epochs (int): Number of training epochs.
            save_epochs (int): Number of epochs at which to save the model checkpoint.
            lr_schedule (Optional[Dict]): Configuration for the learning rate schedule.
            optimizer (Dict): Configuration for the optimizer.
            loss_ramp_up_epochs (int): Number of epochs to ramp up the weight of the 
                unsupervised component of the loss function.
            unsup_loss_weight (float): Weight of the unsupervised loss component.
            alpha (float): Exponential moving average (EMA) coefficient for updating
                the teacher model's weights.
            seed (int): Random seed.
    """

    num_epochs: int = 100
    loss_ramp_up_epochs: int = 50
    unsup_loss_weight: float = 1.0
    alpha: float = 0.99
