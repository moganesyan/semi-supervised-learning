from typing import Union, Optional
from dataclasses import dataclass

@dataclass
class BaseTrainerConfig():
    """
        Base configuration class for all trainers.

        params:
            output_dir (str): Directory where output models and checkpoints
                will be stored.
            batch_size (int): Training batch size.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for parameter updates.
            save_epochs (int): Number of epochs at which to save the model checkpoint.
            seed (int): Random seed.
    """

    output_dir: str = None
    batch_size: int = 100
    num_epochs: int = 100
    learning_rate: float = 1e-4
    save_epochs: Optional[int] = 10
    seed: int = 42
