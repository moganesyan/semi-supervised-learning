from typing import Union, Optional, Dict

from ..base_trainer.base_trainer_config import BaseTrainerConfig


class PiModelTrainerConfig(BaseTrainerConfig):
    """
        Configuration class for the categorical Pi-Model trainer.
        As seen in the original paper: https://arxiv.org/abs/1610.02242

        params:
            output_dir (str): Directory where output models and checkpoints
                will be stored.
            batch_size (int): Training batch size.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for parameter updates.
            save_epochs (int): Number of epochs at which to save the model checkpoint.
            weight_ramp_up_epochs (int): Number of epochs to ramp up the weight of the 
                unsupervised component of the loss function.
            seed (int): Random seed.
    """

    weight_ramp_up_epochs: int = 20
    num_epochs: int = 100
    batch_size: int = 128

    # TODO: should augmentation probabilities go here?