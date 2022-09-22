from typing import Union, Optional, Dict
from dataclasses import dataclass, field


class BaseTrainerConfig():
    """
        Base configuration class for all trainers.

        params:
            output_dir (str): Directory where output models and checkpoints
                will be stored.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for parameter updates.
            save_epochs (int): Number of epochs at which to save the model checkpoint.
            seed (int): Random seed.
    """

    output_dir: str = None
    num_epochs: int = 100
    save_epochs: Optional[int] = 10
    seed: int = 42

    optimizer: Dict = {
        "name": "adam",
        "params": {
            "learning_rate": 1e-4
        }
    }
