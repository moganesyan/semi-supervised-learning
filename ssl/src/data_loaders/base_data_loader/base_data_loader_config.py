from typing import Union, Optional, Dict


class BaseDataLoaderConfig():
    """
        Base configuration class for data loaders.

        All data loader configuration classes should be derived from here.

        The data loader configuration class may encapsulate the following options:

        - output shapes.
        - custom batch options (eg: mixed labelled, unlabelled instances).
        - class balance options.
        - data augmentations.
        - data preprocessing functions (eg: normalisation).

        Note that the above list is not exhaustive.

        No specific configuration style is enforced. It is the responsibility of the
            data loader creator to make sure that the corresponding configuration class
            is appropriate and sufficient for the task.

        params:
            num_classes (int) - Number of classes (unique labels).
            batch_size (int) - Batch size to be used for training and evaluation.
            seed (int) - Random seed.
    """

    num_classes: int = None
    batch_size: int = None
    seed: int = 42
