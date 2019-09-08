import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Conv2D, Concatenate

batch_norm_params = {'decay': 0.9997, 'epsilon': 1e-5}
WEIGHT_DECAY = 0.00004


def batch_norm(inputs: tf.Tensor):
    return BatchNormalization(momentum=batch_norm_params['decay'],
                              epsilon=batch_norm_params['epsilon'])(inputs)


def exit_flow(inputs: tf.Tensor,
              filter_count: int = 256,
              weight_decay: float = WEIGHT_DECAY):
    _x = Conv2D(filter_count,
                kernel_size=1,
                strides=1,
                padding="same",
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(weight_decay))(inputs)
    _x = batch_norm(_x)

    return _x


def dpc_head(inputs: tf.Tensor, weight_decay: float = WEIGHT_DECAY):
    return inputs


def basic_head(inputs: tf.Tensor, weight_decay: float = WEIGHT_DECAY):
    _, width, height, _ = inputs.shape

    left_path = AveragePooling2D(pool_size=(width, height))(inputs)
    left_path = Conv2D(
        256,
        kernel_size=1,
        strides=1,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(weight_decay))(left_path)
    left_path = batch_norm(left_path)
    left_path = tf.image.resize(left_path, [width, height])

    right_path = Conv2D(
        256,
        kernel_size=1,
        strides=1,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(weight_decay))(inputs)
    right_path = batch_norm(right_path)

    _x = Concatenate()([left_path, right_path])
    _x = exit_flow(_x, weight_decay=weight_decay)

    return _x
