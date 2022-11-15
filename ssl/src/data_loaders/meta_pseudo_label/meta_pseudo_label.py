from typing import Union, Optional, Dict, Callable, Tuple, Union

import tensorflow as tf

from ..base_data_loader.base_data_loader import BaseDataLoader
from .meta_pseudo_label_config import MetaPseudoLabelDataLoaderConfig

from ...augmenters.affine import apply_crop_and_resize
from ...augmenters.colour import apply_colour_distortion
from ...augmenters.blur import apply_gaussian_blur


class MetaPseudoLabelDataLoader(BaseDataLoader):
    """
        Class for the Meta Pseudo Label data loader.

        args:
            data_in (tf.data.Dataset) - Dataset object.
            data_loader_config (MetaPseudoLabelDataLoaderConfig): Configuration class for the data loader.
        returns:
            None
    """

    def __init__(self,
                 data_in: tf.data.Dataset,
                 data_loader_config: MetaPseudoLabelDataLoaderConfig) -> None:
        super().__init__(data_loader_config)
        self._data_in = data_in

    def _get_preprocessing_func_train(self) -> Callable:
        """
            Get function that applies training preprocessing steps.

            1) Scaling features values between 0 and 1.

            args:
                None
            returns:
                preproc_func (Callable) - Data preprocessing function.
        """

        def preproc_func(features: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            """
                Apply training preprocessing steps.

                This function is meant to be applied elementwise.

                args:
                    features (tf.Tensor) - Features.
                    label (tf.Tensor) - Label(s).
                return:
                    features_proc (tf.Tensor) - Preprocessed features.
                    label (tf.Tensor) - Labels(s).
            """

            features_proc = tf.cast(features, tf.float32) / 255.
            return features_proc, label

        return preproc_func

    def _get_preprocessing_func_inf(self) -> Callable:
        """
            Get function that applies inference preprocessing steps.

            1) Scale feature values between 0 and 1.
            2) Apply one-hot encoding on labelled samples.

            args:
                None
            returns:
                preproc_func (Callable) - Data preprocessing function.
        """

        num_classes = self._data_loader_config.num_classes

        def preproc_func(features: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            """
                Apply inference preprocessing steps.

                This function is meant to be applied elementwise.

                args:
                    features (tf.Tensor) - Features.
                    label (tf.Tensor) - Label(s).
                return:
                    features_proc (tf.Tensor) - Preprocessed features.
                    label_onehot (tf.Tensor) - One-hot encoded labels(s).
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

        blur_chance = (self._data_loader_config.blur_params['chance']
            if self._data_loader_config.blur_params is not None else 0.0)
        crop_chance = (self._data_loader_config.crop_params['chance']
            if self._data_loader_config.crop_params is not None else 0.0)
        jitter_chance = (self._data_loader_config.jitter_params['chance']
            if self._data_loader_config.jitter_params is not None else 0.0)

        def aug_func(features: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            """
                Apply two realisations of random data augmentations.

                This function is meant to be applied elementwise.

                args:
                    features (tf.Tensor) - Features.
                    label (tf.Tensor) - Label(s).
                return:
                    features_aug (tf.Tensor) - Augmented features.
                    label (tf.Tensor) - Labels(s).
            """

            features_aug = features[tf.newaxis, ...]

            # roll augmentation chances
            roll_blur = tf.random.uniform((), 0, 1.0, dtype = tf.float32)
            roll_crop = tf.random.uniform((), 0, 1.0, dtype = tf.float32)
            roll_jitter = tf.random.uniform((), 0, 1.0, dtype = tf.float32)

            # apply random augmentations

            # apply random gaussian blur
            if blur_chance != 0:
                if roll_blur <= blur_chance:
                    features_aug = apply_gaussian_blur(
                        features_aug,
                        **self._data_loader_config.blur_params
                    )

            # apply random crop and resize
            if crop_chance != 0:
                if roll_crop <= crop_chance:
                    features_aug = apply_crop_and_resize(
                        features_aug,
                        **self._data_loader_config.crop_params
                    )

            # apply random colour jitter / distortion
            if jitter_chance != 0:
                if roll_jitter <= jitter_chance:
                    features_aug = apply_colour_distortion(
                        features_aug,
                        **self._data_loader_config.jitter_params
                    )

            return tf.squeeze(features_aug), label

        return aug_func

    def _get_batch_func(self) -> Callable:
        """
            Get custom batching function.

            1) Mask unlabelled samples.
            2) Apply one-hot encoding on labelled samples.

            args:
                None
            returns:
                batch_func (Callable) - Custom batching function.
        """

        num_classes = self._data_loader_config.num_classes

        def batch_func(features_batch: tf.Tensor,
                       labels_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            """
                Apply custom batching logic.

                Batch is split into labelled and unlabelled minibatches.
                Unlabelled instances are identified by a class label of -1.

                This function is meant to be applied batchwise.

                args:
                    features_batch (tf.Tensor) - Batch of augmented features.
                    labels_batch (tf.Tensor) - Batch of labels.
                return:
                    features_batch (tf.Tensor) - Batch of augmented features.
                    label_batch_onehot (tf.Tensor) - Batch of one-hot encoded class labels.
                    mask_batch (tf.Tensor) - Batch of flags indicating labelled samples.
            """

            # get mask batch
            mask_batch = tf.not_equal(labels_batch, -1)

            # change the placeholder -1 labels to 0 (a valid class label)
            labels_batch = labels_batch * tf.cast(mask_batch, tf.int64)

            # encode labels
            label_batch_onehot = tf.squeeze(tf.one_hot(labels_batch, num_classes))

            return features_batch, label_batch_onehot, mask_batch

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
            preproc_func = self._get_preprocessing_func_train()
            aug_func = self._get_augmentation_func()
            batch_func = self._get_batch_func()

            dataset = self._data_in
            dataset = dataset.shuffle(buffer_size=self._data_loader_config.shuffle_buffer_size)
            dataset = dataset.map(preproc_func)
            dataset = dataset.map(aug_func)
            dataset = dataset.batch(self._data_loader_config.batch_size)
            dataset = dataset.map(batch_func)
        else:
            preproc_func = self._get_preprocessing_func_inf()

            # chain operators
            dataset = self._data_in
            dataset = dataset.map(preproc_func)
            dataset = dataset.batch(self._data_loader_config.batch_size)

        return dataset
