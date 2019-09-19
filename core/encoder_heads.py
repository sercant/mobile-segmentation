import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Conv2D, Concatenate, DepthwiseConv2D

# This is to fix the bug of https://github.com/tensorflow/tensorflow/issues/27298
# if it gets fixed remove the related lines
from tensorflow.python.keras.backend import get_graph

BATCH_NORM_PARAMS = {'decay': 0.9997, 'epsilon': 1e-5}
WEIGHT_DECAY = 0.00004


def batch_norm(inputs: tf.Tensor):
    _x = BatchNormalization(momentum=BATCH_NORM_PARAMS['decay'],
                            epsilon=BATCH_NORM_PARAMS['epsilon'])(inputs)

    return _x


def exit_flow(inputs: tf.Tensor,
              filter_count: int = 256,
              weight_decay: float = WEIGHT_DECAY):
    with tf.name_scope("exit_flow"):
        _x = Conv2D(
            filter_count,
            kernel_size=1,
            strides=1,
            padding="same",
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(weight_decay))(inputs)
        _x = batch_norm(_x)

    return _x


def dpc_head(inputs: tf.Tensor,
             weight_decay: float = WEIGHT_DECAY,
             filter_per_branch: int = 256):
    def dpc_conv_op(inputs: tf.Tensor,
                    rate: list,
                    weight_decay: float = WEIGHT_DECAY):
        _x = DepthwiseConv2D(kernel_size=3,
                             strides=1,
                             dilation_rate=rate,
                             activation="relu",
                             padding="same")(inputs)
        _x = batch_norm(_x)
        _x = Conv2D(filter_per_branch,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(weight_decay))(_x)
        _x = batch_norm(_x)

        return _x

    with tf.name_scope("dpc_head"):
        # depth 1
        _x = dpc_conv_op(inputs, rate=[1, 6], weight_decay=weight_decay)

        # depth 2
        _x1 = dpc_conv_op(_x, rate=[18, 15], weight_decay=weight_decay)
        _x2 = dpc_conv_op(_x, rate=[6, 21], weight_decay=weight_decay)
        _x3 = dpc_conv_op(_x, rate=[1, 1], weight_decay=weight_decay)

        # depth 3
        _x4 = dpc_conv_op(_x1, rate=[6, 3], weight_decay=weight_decay)

        _x = Concatenate()([_x, _x1, _x2, _x3, _x4])

    _x = exit_flow(_x,
                   filter_count=filter_per_branch,
                   weight_decay=weight_decay)

    return _x


def basic_head(inputs: tf.Tensor,
               weight_decay: float = WEIGHT_DECAY,
               filter_per_branch: int = 256):
    _, width, height, _ = inputs.shape

    with tf.name_scope("basic_head"):
        left_path = AveragePooling2D(pool_size=(width, height))(inputs)
        left_path = Conv2D(
            filter_per_branch,
            kernel_size=1,
            strides=1,
            padding="same",
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(weight_decay))(left_path)
        left_path = batch_norm(left_path)
        left_path = tf.image.resize(left_path, [width, height])

        right_path = Conv2D(
            filter_per_branch,
            kernel_size=1,
            strides=1,
            padding="same",
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(weight_decay))(inputs)
        right_path = batch_norm(right_path)

        _x = Concatenate()([left_path, right_path])
    _x = exit_flow(_x,
                   filter_count=filter_per_branch,
                   weight_decay=weight_decay)

    return _x
