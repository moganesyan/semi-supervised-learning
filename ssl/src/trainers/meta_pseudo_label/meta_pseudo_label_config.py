from typing import Union, Optional, Dict

from ..base_trainer.base_trainer_config import BaseTrainerConfig


class MetaPseudoLabelTrainerConfig(BaseTrainerConfig):
    """
        Configuration class for the Meta Pseudo Label trainer.
        As seen in the original paper: https://arxiv.org/abs/2003.10580

        params:
            output_dir (str): Directory where output models and checkpoints
                will be stored.
            num_epochs (int): Number of training epochs.
            num_epochs_finetune (int): Number of epochs to finetune the model.
            save_epochs (int): Number of epochs at which to save the model checkpoint.
            lr_schedule (Optional[Dict]): Configuration for the learning rate schedule.
            optimizer (Dict): Configuration for the optimizer.
            uda_loss_weight (float): Weight factor for the UDA loss component.
            uda_conf_thresh (float): Confidence threshold for the UDA loss component to count.
            uda_softmax_temp (float): Softmax temperature for pseudo label generation.
            seed (int): Random seed.
    """

    num_epochs: int = 100
    num_epochs_finetune: int = 50

    uda_loss_weight: float = 1.0
    uda_conf_thresh: float = 0.80
    uda_softmax_temp: float = 0.40
