from typing import Union, Optional, Dict, Callable, Tuple, Union

import numpy as np
import tensorflow as tf

from ..pi_model_data_loader.pi_model_data_loader_config import PiModelDataLoaderConfig
from ...augmenters.affine import apply_crop_and_resize
from ...augmenters.colour import apply_colour_distortion
from ...augmenters.blur import apply_gaussian_blur


class PiModelDataLoader():
    """
        Class for the Pi-Model data loader.

        args:
            data_in (Union[tf.data.Dataset, Tuple[tf.data.Dataset, tf.data.Dataset]]):
                Tuple of labelled and unlabelled dataset if training, or a test dataset.
            data_loader_config (PiModelDataLoaderConfig): Configuration class for the data loader.
        returns:
            None
    """

    def __init__(self,
                 data_in: Union[tf.data.Dataset, Tuple[tf.data.Dataset, tf.data.Dataset]],
                 data_loader_config: PiModelDataLoaderConfig) -> None:
        self._data_in = data_in
        self._data_loader_config = data_loader_config

    def _get_preprocessing_func(self) -> Callable:
        """
            Get function that applies data preprocessing steps.

            1) Scaling image values between 0 and 1.

            args:
                None
            returns:
                preproc_func (Callable) - Data preprocessing function.
        """

        def preproc_func(features: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            """
                Apply data preprocessing steps.

                This function is meant to be applied elementwise.

                args:
                    features (tf.Tensor) - Features on which to apply preprocessing steps.
                    label (tf.Tensor) - Label(s) for the sample instance.
                return:
                    features_proc (tf.Tensor) - Features after preprocessing steps have been applied.
                    label (tf.Tensor) - Labels(s) for the sample instance.
            """

            features_proc = tf.cast(features, tf.float32) / 255.
            return features_proc, label

        preproc_func

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

            features_aug = features

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

            return features_aug, label

        return aug_func

    def _get_batch_func(self) -> Callable:
        """
            Get custom batching function.

            1) Split batch into labelled and unlabelled samples.
            2) Apply one-hot encoding on labelled samples.

            args:
                None
            returns:
                batch_func (Callable) - Custom batching function.
        """

        num_classes = self._data_loader_config.num_classes

        def batch_func(features_batch: tf.Tensor, labels_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            """
                Apply custom batching logic.

                Batch is split into labelled and unlabelled minibatches.
                Unlabelled instances are identified by a class label of -1.

                This function is meant to be applied batchwise.

                args:
                    features_batch (tf.Tensor) - Batch of features.
                    labels_batch (tf.Tensor) - Batch of labels.
                return:
                    features_batch_labelled (tf.Tensor) - Batch of features with labels.
                    features_batch_unlabelled (tf.Tensor) - Batch of features without labels.
                    label_batch_onehot (tf.Tensor) - Batch of one-hot encoded class labels.
            """

            # idx for unlabelled data
            idx_labelled = tf.where(tf.not_equal(labels_batch, -1))[:,0]
            idx_unlabelled = tf.where(tf.equal(labels_batch, -1))[:,0]

            # split batch
            features_batch_labelled = tf.gather(features_batch, indices = idx_labelled, axis = 0)
            labels_batch_labelled = tf.gather(labels_batch, indices = idx_labelled, axis = 0)

            features_batch_unlabelled = tf.gather(features_batch, indices = idx_unlabelled, axis = 0)

            # encode labels
            label_batch_onehot = tf.squeeze(tf.one_hot(labels_batch_labelled, num_classes))

            return features_batch_labelled, features_batch_unlabelled, label_batch_onehot

        return batch_func

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
            batch_func = self._get_batch_func

            # chain operators
            dataset = tf.data.Dataset.sample_from_datasets(
                [self._data_in[0], self._data_in[1]],
                self._data_loader_config.batch_ratios
            )
            dataset = dataset.shuffle(buffer_size=self._data_loader_config.shuffle_buffer_size)
            dataset = dataset.map(preproc_func)
            dataset = dataset.map(aug_func)
            dataset = dataset.batch(self._data_loader_config.batch_size)
            dataset = dataset.map(batch_func)
        else:
            preproc_func = self._get_preprocessing_func()

            # chain operators
            dataset = self._data_in
            dataset = dataset.map(batch_func)

        return dataset

