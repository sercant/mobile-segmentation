import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, BatchNormalization, MaxPool2D,
                                     DepthwiseConv2D, Concatenate, Reshape,
                                     Permute, Concatenate, Lambda)

# This is to fix the bug of https://github.com/tensorflow/tensorflow/issues/27298
# if it gets fixed remove the related lines
from tensorflow.python.keras.backend import get_graph

BATCH_NORM_PARAMS = {'decay': 0.9997, 'epsilon': 1e-3}
WEIGHT_DECAY = 0.00004


def batch_norm(inputs: tf.Tensor):
    _x = BatchNormalization(momentum=BATCH_NORM_PARAMS['decay'],
                            epsilon=BATCH_NORM_PARAMS['epsilon'])(inputs)

    return _x


def entry_layer(inputs: tf.Tensor, weight_decay: float = WEIGHT_DECAY):
    with tf.name_scope("entry_layer"):
        _x = Conv2D(
            24,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(weight_decay))(inputs)
        _x = batch_norm(_x)
        _x = MaxPool2D(padding="same")(_x)

    return _x


def basic_unit_with_downsampling(inputs: tf.Tensor,
                                 out_channels: int = None,
                                 stride: int = 2,
                                 rate: int = 1,
                                 weight_decay: float = WEIGHT_DECAY):
    in_channels = inputs.shape[-1]
    out_channels = 2 * in_channels if out_channels is None else out_channels

    # right path
    right_path = Conv2D(
        in_channels,
        kernel_size=1,
        strides=1,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(weight_decay))(inputs)
    right_path = batch_norm(right_path)
    right_path = DepthwiseConv2D(kernel_size=3,
                                 strides=stride,
                                 dilation_rate=rate,
                                 padding="same")(right_path)
    right_path = batch_norm(right_path)
    right_path = Conv2D(
        out_channels // 2,
        kernel_size=1,
        strides=1,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(weight_decay))(right_path)
    right_path = batch_norm(right_path)

    # left path
    left_path = DepthwiseConv2D(kernel_size=3,
                                strides=stride,
                                dilation_rate=rate,
                                padding="same")(inputs)
    left_path = batch_norm(left_path)
    left_path = Conv2D(
        out_channels // 2,
        kernel_size=1,
        strides=1,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(weight_decay))(left_path)
    left_path = batch_norm(left_path)

    _x = concat_shuffle([left_path, right_path])

    return _x


def channel_shuffle(inputs: tf.Tensor):
    _, height, width, depth = inputs.shape

    _x = Reshape([-1, 2, depth // 2])(inputs)
    _x = Permute([1, 3, 2])(_x)
    _x = Reshape([height, width, depth])(_x)

    return _x


def concat_shuffle(inputs: list):
    _x = Concatenate()(inputs)
    _x = channel_shuffle(_x)

    return _x


def basic_unit(inputs: tf.Tensor,
               rate: int = 1,
               weight_decay: float = WEIGHT_DECAY):
    splits = tf.split(inputs, num_or_size_splits=2, axis=3)

    in_channels = splits[0].shape[-1]

    _x = Conv2D(in_channels,
                kernel_size=1,
                strides=1,
                padding="same",
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(weight_decay))(
                    splits[1])
    _x = batch_norm(_x)
    _x = DepthwiseConv2D(kernel_size=3,
                         strides=1,
                         dilation_rate=rate,
                         padding="same")(_x)
    _x = Conv2D(in_channels,
                kernel_size=1,
                strides=1,
                padding="same",
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(weight_decay))(_x)
    _x = batch_norm(_x)

    _x = concat_shuffle([splits[0], _x])

    return _x


def shufflenet_v2_base(inputs: tf.Tensor,
                       depth_multiplier: float,
                       output_stride: int = 32,
                       weight_decay: float = WEIGHT_DECAY):
    depth_multipliers = {0.5: 48, 1.0: 116, 1.5: 176, 2.0: 224}
    initial_depth = depth_multipliers[depth_multiplier]

    if output_stride < 4:
        raise ValueError("Output stride should be cannot be lower than 4.")

    layer_info = [
        {
            "num_units": 3,
            "out_channels": initial_depth,
            "scope": "stage_2",
            "stride": 2
        },
        {
            "num_units": 7,
            "out_channels": None,
            "scope": "stage_3",
            "stride": 2
        },
        {
            "num_units": 3,
            "out_channels": None,
            "scope": "stage_4",
            "stride": 2
        },
    ]

    def stride_handling(stride: int, current_stride: int, current_rate: int,
                        max_stride: int):
        if current_stride == max_stride:
            return 1, current_rate * stride
        else:
            current_stride *= stride
            return stride, current_rate

    with get_graph().as_default(), tf.name_scope("shufflenet_v2"):
        _x = entry_layer(inputs, weight_decay)

        current_stride = 4
        current_rate = 1
        for i in range(3):
            layer = layer_info[i]
            stride, rate = stride_handling(layer["stride"], current_stride,
                                           current_rate, output_stride)

            with tf.name_scope(layer["scope"]):
                _x = basic_unit_with_downsampling(_x,
                                                  layer["out_channels"],
                                                  stride=stride,
                                                  rate=rate,
                                                  weight_decay=weight_decay)
                for _ in range(layer["num_units"]):
                    _x = basic_unit(_x, rate=rate, weight_decay=weight_decay)

            current_stride *= stride
            current_rate *= rate

    return _x


def shufflenet_v2(inputs: tf.Tensor,
                  num_classes: int,
                  depth_multiplier: float = 1.0,
                  output_stride: int = 32,
                  weight_decay: float = WEIGHT_DECAY):
    from tensorflow.keras import layers

    _x = shufflenet_v2_base(inputs, depth_multiplier, output_stride)

    final_channels = 1024 if depth_multiplier != "2.0" else 2048

    with get_graph().as_default(), tf.name_scope("shufflenet_v2/logits"):
        _x = Conv2D(final_channels,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    kernel_regularizer=keras.regularizers.l2(weight_decay))(_x)
        _x = layers.GlobalAveragePooling2D()(_x)
        _x = layers.Dense(num_classes,
                          activation="softmax",
                          kernel_initializer="he_normal")(_x)

    return _x


if __name__ == "__main__":
    # tf.random.set_seed(22)

    # # The data, split between train and test sets:
    # (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    # print("x_train shape:", x_train.shape)
    # print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")

    # # Convert class vectors to binary class matrices.
    # y_train = keras.utils.to_categorical(y_train, 10)
    # y_test = keras.utils.to_categorical(y_test, 10)

    # x_train = x_train.astype("float32")
    # x_test = x_test.astype("float32")
    # x_train /= 255
    # x_test /= 255

    # input_shape = (32, 32, 3)
    # inputs = keras.Input(shape=input_shape)

    # output = shufflenet_v2(inputs, 10, 1.0)
    # model = tf.keras.Model(inputs=inputs, outputs=output)

    # model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # model.fit(
    #     x_train,
    #     y_train,
    #     batch_size=128,
    #     epochs=10,
    #     validation_data=(x_test, y_test),
    #     shuffle=True,
    # )

    # scores = model.evaluate(x_test, y_test, verbose=1)
    # print("Test loss:", scores[0])
    # print("Test accuracy:", scores[1])

    # model.save("model.h5")

    inputs = keras.Input(shape=(32, 32, 3))
    output = shufflenet_v2(inputs, 10, 1.0)

    model = keras.Model(inputs=inputs, outputs=output)
    model.load_weights('./checkpoints/cifar10.h5')

    model.summary()
