from typing import Union, Optional, Dict

from ..base_trainer.base_trainer_config import BaseTrainerConfig


class PseudoLabelTrainerConfig(BaseTrainerConfig):
    """
        Configuration class for the Pseudo Model trainer.
        As seen in the original paper: https://www.researchgate.net/publication/280581078_Pseudo-Label_The_Simple_and_Efficient_Semi-Supervised_Learning_Method_for_Deep_Neural_Networks

        params:
            output_dir (str): Directory where output models and checkpoints
                will be stored.
            num_epochs (int): Number of training epochs.
            save_epochs (int): Number of epochs at which to save the model checkpoint.
            lr_schedule (Optional[Dict]): Configuration for the learning rate schedule.
            optimizer (Dict): Configuration for the optimizer.
            t1 (int): Number of epochs for which unsupervised loss component is unused.
            t1 (int): Number of epochs for which unsupervised loss component is annealed towards alpha.
            alpha (float): Weight coefficient for unsupervised loss component.
            seed (int): Random seed.
    """

    num_epochs: int = 100
    t1: int = 25
    t2: int = 50
    alpha: float = 1.0
