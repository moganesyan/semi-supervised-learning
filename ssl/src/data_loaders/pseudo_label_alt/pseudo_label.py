from typing import Union, Optional, Dict, Callable, Tuple, Union

import tensorflow as tf

from ..base_data_loader.base_data_loader import BaseDataLoader
from .pseudo_label_config import PseudoLabelDataLoaderConfig


class PseudoLabelDataLoader(BaseDataLoader):
    """
        Class for the Pseudo Label data loader.

        args:
            data_in (tf.data.Dataset) - Dataset object.
            data_loader_config (PiModelDataLoaderConfig): Configuration class for the data loader.
        returns:
            None
    """

    def __init__(self,
                 data_in: tf.data.Dataset,
                 data_loader_config: PseudoLabelDataLoaderConfig) -> None:
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
                    features (tf.Tensor) - Features on which to apply preprocessing steps.
                    label (tf.Tensor) - Label(s) for the sample instance.
                return:
                    features_proc (tf.Tensor) - Features after preprocessing steps have been applied.
                    label_proc (tf.Tensor) - Labels(s) for the sample instance.
            """

            features_proc = tf.cast(features, tf.float32) / 255.
            label_proc = tf.cast(label, tf.int32)
            return features_proc, label_proc

        return preproc_func

    def _get_preprocessing_func_inf(self) -> Callable:
        """
            Get function that applies inference preprocessing steps.

            1) Scaling features values between 0 and 1.
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
                    features (tf.Tensor) - Features on which to apply preprocessing steps.
                    label (tf.Tensor) - Label(s) for the sample instance.
                return:
                    features_proc (tf.Tensor) - Features after preprocessing steps have been applied.
                    label (tf.Tensor) - One-hot encoded labels(s) for the sample instance.
            """

            features_proc = tf.cast(features, tf.float32) / 255.
            label_onehot = tf.squeeze(tf.one_hot(label, num_classes))

            return features_proc, label_onehot

        return preproc_func

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

        def batch_func(features_batch: tf.Tensor,
                        labels_batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            """
                Apply custom batching logic.

                Batch consists of labelled and unlabelled minibatches.
                Unlabelled instances are identified by a class label of -1.

                This function is meant to be applied batchwise.

                args:
                    features_batch (tf.Tensor) - Batch of input features.
                    labels_batch (tf.Tensor) - Batch of class labels.
                return:
                    features_batch (tf.Tensor) - Batch of labelled features.
                    label_batch_onehot (tf.Tensor) - Batch of one-hot encoded class labels.
                    mask_batch (tf.Tensor) - Batch of flags indicating labelled samples.
            """

            # get mask batch
            mask_batch = tf.not_equal(labels_batch, -1)

            # change the placeholder -1 labels to 0 (a valid class label)
            labels_batch = labels_batch * tf.cast(mask_batch, tf.int32)

            # encode labels
            label_batch_onehot = tf.squeeze(tf.one_hot(labels_batch, num_classes))

            return features_batch, label_batch_onehot, mask_batch

        return batch_func

    def _build_data_loader(self, training: bool) -> tf.data.Dataset:
        """
            Build data loader from the configuration object.

            1) Apply data preprocessing functions.
            2) Apply custom batching function.

            args:
                training: bool - Flag to indicate if the dataset is for training.
            returns:
                dataset (tf.data.Dataset) - Dataset iterable.
        """

        if training:
            preproc_func = self._get_preprocessing_func_train()
            batch_func = self._get_batch_func()

            dataset = self._data_in
            dataset = dataset.shuffle(buffer_size=self._data_loader_config.shuffle_buffer_size)
            dataset = dataset.map(preproc_func)
            dataset = dataset.batch(self._data_loader_config.batch_size)
            dataset = dataset.map(batch_func)
        else:
            preproc_func = self._get_preprocessing_func_inf()

            # chain operators
            dataset = self._data_in
            dataset = dataset.map(preproc_func)
            dataset = dataset.batch(self._data_loader_config.batch_size)

        return dataset
