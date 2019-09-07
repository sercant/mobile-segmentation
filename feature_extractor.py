#%%
import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.python import keras
from tensorflow.python.keras import Model, layers
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPool2D,
    DepthwiseConv2D,
    Concatenate,
    Reshape,
    Permute,
    Concatenate,
    Lambda,
)

BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3
WEIGHT_DECAY = 0.00004


def batch_norm(inputs: tf.Tensor):
    # batch_norm = keras.Sequential()
    # batch_norm.add(relu)
    # batch_norm.add(batch_norm)

    return BatchNormalization(momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPSILON)(
        inputs
    )


def entry_layer(inputs: tf.Tensor):
    # entry
    x = Conv2D(
        24,
        kernel_size=3,
        strides=2,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(inputs)
    x = batch_norm(x)
    x = MaxPool2D(padding="same")(x)

    return x


def basic_unit_with_downsampling(
    inputs: tf.Tensor, out_channels: int = None, stride: int = 2, rate: int = 1
):
    in_channels = inputs.shape[-1]
    out_channels = 2 * in_channels if out_channels is None else out_channels

    # right path
    right_path = Conv2D(
        in_channels,
        kernel_size=1,
        strides=1,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(inputs)
    right_path = batch_norm(right_path)
    right_path = DepthwiseConv2D(
        kernel_size=3, strides=stride, dilation_rate=rate, padding="same"
    )(right_path)
    right_path = batch_norm(right_path)
    right_path = Conv2D(
        out_channels // 2,
        kernel_size=1,
        strides=1,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(right_path)
    right_path = batch_norm(right_path)

    # left path
    left_path = DepthwiseConv2D(
        kernel_size=3, strides=stride, dilation_rate=rate, padding="same"
    )(inputs)
    left_path = batch_norm(left_path)
    left_path = Conv2D(
        out_channels // 2,
        kernel_size=1,
        strides=1,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(left_path)
    left_path = batch_norm(left_path)

    x = concat_shuffle([left_path, right_path])

    return x


def channel_shuffle(inputs: tf.Tensor):
    _, height, width, depth = inputs.shape

    x = Reshape([-1, 2, depth // 2])(inputs)
    x = Permute([1, 3, 2])(x)
    x = Reshape([height, width, depth])(x)

    return x


def concat_shuffle(inputs: list):
    x = Concatenate()(inputs)
    x = channel_shuffle(x)

    return x


def basic_unit(inputs: tf.Tensor, rate: int = 1):
    splits = tf.split(inputs, num_or_size_splits=2, axis=3)

    in_channels = splits[0].shape[-1]

    x = Conv2D(
        in_channels,
        kernel_size=1,
        strides=1,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(splits[1])
    x = batch_norm(x)
    x = DepthwiseConv2D(kernel_size=3, strides=1, dilation_rate=rate, padding="same")(x)
    x = Conv2D(
        in_channels,
        kernel_size=1,
        strides=1,
        padding="same",
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(x)
    x = batch_norm(x)

    x = concat_shuffle([splits[0], x])

    return x


def shufflenet_v2_base(inputs: tf.Tensor, depth_multiplier: float):
    depth_multipliers = {0.5: 48, 1.0: 116, 1.5: 176, 2.0: 224}
    initial_depth = depth_multipliers[depth_multiplier]

    x = entry_layer(inputs)

    layer_info = [
        {"num_units": 4, "out_channels": initial_depth, "scope": "Stage2", "stride": 2},
        {"num_units": 8, "out_channels": None, "scope": "Stage3", "stride": 2},
        {"num_units": 4, "out_channels": None, "scope": "Stage4", "stride": 2},
    ]

    for i in range(3):
        layer = layer_info[i]
        x = basic_unit_with_downsampling(x, layer["out_channels"])
        for j in range(2, layer["num_units"] + 1):
            x = basic_unit(x)

    return x


def shufflenet_v2(inputs: tf.Tensor, num_classes: int, depth_multiplier: float):
    x = shufflenet_v2_base(inputs, depth_multiplier)

    final_channels = 1024 if depth_multiplier != "2.0" else 2048

    x = Conv2D(
        final_channels,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_regularizer=keras.regularizers.l2(WEIGHT_DECAY),
    )(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(
        x
    )

    return x


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

    # model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

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

    inputs = keras.Input(shape=(224, 224, 3))
    output = shufflenet_v2(inputs, 1000, 1.0)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.summary()
    model.save("shufflenet_v2.h5")
