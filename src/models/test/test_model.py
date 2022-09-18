import tensorflow as tf


class TestModel():
    def __init__(self, input_shape, num_classes) -> None:
        self._input_shape = input_shape
        self._num_classes = num_classes

    def _build_network(self) -> tf.keras.Model:

        x_in = tf.keras.Input(shape=self._input_shape, name="input")
        x = tf.identity(x_in)

        x = tf.keras.layers.Conv2D(
            32, kernel_size = 3)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.GlobalMaxPooling2D()(x)
        x = tf.keras.layers.Dropout(0.35)(x)

        x = tf.keras.layers.Dense(100)(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Dense(self._num_classes)(x)
        x_out = tf.keras.layers.Softmax()(x)

        model = tf.keras.Model(inputs=x_in, outputs=x_out)
        return model

    def __call__(self) -> tf.keras.Model:
        return self._build_network()