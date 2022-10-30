from typing import Union, Optional, Dict, List
from dataclasses import dataclass, field


class BaseTrainerConfig():
    """
        Base configuration class for all trainers.

        params:
            output_dir (str): Directory where output models and checkpoints
                will be stored.
            num_epochs (int): Number of training epochs.
            save_epochs (int): Number of epochs at which to save the model checkpoint.
            lr_schedule (Optional[Dict]): Configuration for the learning rate schedule.
            optimizer (Dict): Configuration for the optimizer.
            callbacks (Optional[List[Dict]]): List of callbacks and their configurations.
            seed (int): Random seed.
    """

    output_dir: str = None
    num_epochs: int = 100
    save_epochs: Optional[int] = 10

    lr_schedule: Optional[Dict] = None
    optimizer: Dict = {
        "name": "adam",
        "learning_rate": 1e-4,
        "params": {
        }
    }
    callbacks: Optional[List[Dict]] = None

    seed: int = 42
