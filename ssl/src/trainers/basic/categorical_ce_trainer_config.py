from typing import Union, Optional, Dict

from ..base_trainer.base_trainer_config import BaseTrainerConfig


class CategoricalCETrainerConfig(BaseTrainerConfig):
    """
        Configuration class for the categorical CE trainer.

        params:
            output_dir (str): Directory where output models and checkpoints
                will be stored.
            batch_size (int): Training batch size.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for parameter updates.
            save_epochs (int): Number of epochs at which to save the model checkpoint.
            seed (int): Random seed.
    """

    pass