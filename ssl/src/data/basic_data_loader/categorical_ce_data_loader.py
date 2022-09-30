from typing import Union, Optional, Dict, Callable, Tuple, Union

import tensorflow as tf

from ..base_data_loader.base_data_loader import BaseDataLoader
from .categorical_ce_data_loader_config import CategoricalCEDataLoaderConfig

from ...augmenters.affine import apply_crop_and_resize
from ...augmenters.colour import apply_colour_distortion
from ...augmenters.blur import apply_gaussian_blur


class CategoricalCEDataLoader(BaseDataLoader):
    """
        Class for the categorical CE data loader.

        args:
            data_in (tf.data.Dataset): Labelled dataset.
            data_loader_config (CategoricalCEDataLoaderConfig): Configuration class for the data loader.
        returns:
            None
    """

    def __init__(self,
                 data_in: tf.data.Dataset,
                 data_loader_config: CategoricalCEDataLoaderConfig) -> None:
        super().__init__(data_loader_config)
        self._data_in = data_in

    def _get_preprocessing_func(self) -> Callable:
        """
            Get function that applies data preprocessing steps.

            1) Scaling image values between 0 and 1.
            2) One-hot encode label.

            args:
                None
            returns:
                preproc_func (Callable) - Data preprocessing function.
        """

        num_classes = self._data_loader_config.num_classes

        def preproc_func(features: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            """
                Apply data preprocessing steps.

                This function is meant to be applied elementwise.

                args:
                    features (tf.Tensor) - Features on which to apply preprocessing steps.
                    label (tf.Tensor) - Label(s) for the sample instance.
                return:
                    features_proc (tf.Tensor) - Features after preprocessing steps have been applied.
                    label_onehot (tf.Tensor) - One-hot encoded labels(s) for the sample instance.
            """

            features_proc = tf.cast(features, tf.float32) / 255.
            label_onehot = tf.squeeze(tf.one_hot(label, num_classes))

            return features_proc, label_onehot

        return preproc_func

    def _get_augmentation_func(self) -> Callable:
        """
            Get function that applies data augmentations.

            Probability of applying each augmentor is
                defined in the data loader config object.

            1) Random gaussian blur.
            2) Random crop & resize.
            3) Random colour jitter.

            args:
                None
            returns:
                aug_func (Callable) - Data augmentation function.
        """

        blur_chance = self._data_loader_config.blur_chance
        crop_chance = self._data_loader_config.crop_chance
        jitter_chance = self._data_loader_config.jitter_chance

        def aug_func(features: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            """
                Apply random data augmentations.

                This function is meant to be applied elementwise.

                args:
                    features (tf.Tensor) - Features on which to apply random augmentations.
                    label (tf.Tensor) - Label(s).
                return:
                    features_aug (tf.Tensor) - Features after random augmentations have been applied.
                    label (tf.Tensor) - Labels(s).
            """

            features_aug = features[tf.newaxis, ...]

            roll_blur = tf.random.uniform((), 0, 1.0, dtype = tf.float32)
            roll_crop = tf.random.uniform((), 0, 1.0, dtype = tf.float32)
            roll_jitter = tf.random.uniform((), 0, 1.0, dtype = tf.float32)

            # apply random gaussian blur
            if roll_blur <= blur_chance:
                features_aug = apply_gaussian_blur(
                    features_aug
                )

            # apply random crop and resize
            if roll_crop <= crop_chance:
                features_aug = apply_crop_and_resize(
                    features_aug
                )

            # apply random colour jitter / distortion
            if roll_jitter <= jitter_chance:
                features_aug = apply_colour_distortion(
                    features_aug
                )

            return tf.squeeze(features_aug), label

        return aug_func

    def _build_data_loader(self, training: bool) -> tf.data.Dataset:
        """
            Build data loader from the configuration object.

            1) Apply data preprocessing functions.
            2) Apply random data augmentations.
            3) Apply custom batching function.

            args:
                training: bool - Flag to indicate if the dataset is for training.
            returns:
                dataset (tf.data.Dataset) - Dataset iterable.
        """

        if training:
            preproc_func = self._get_preprocessing_func()
            aug_func = self._get_augmentation_func()

            # chain operators
            dataset = self._data_in
            dataset = dataset.shuffle(buffer_size=self._data_loader_config.shuffle_buffer_size)
            dataset = dataset.map(preproc_func)
            dataset = dataset.map(aug_func)
            dataset = dataset.batch(self._data_loader_config.batch_size)
        else:
            preproc_func = self._get_preprocessing_func()

            # chain operators
            dataset = self._data_in
            dataset = dataset.map(preproc_func)
            dataset = dataset.batch(self._data_loader_config.batch_size)

        return dataset
