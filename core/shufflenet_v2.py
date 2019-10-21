import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Conv2D,
                                     DepthwiseConv2D, BatchNormalization, ReLU,
                                     Reshape, Permute, Concatenate, Multiply)
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers


def hard_sigmoid(inputs: tf.Tensor):
    return K.relu(inputs + 3.0, max_value=6.0) / 6.0


class HardSwish(layers.Layer):
    def __init__(self):
        super(HardSwish, self).__init__()

    def call(self, inputs: tf.Tensor):
        return inputs * hard_sigmoid(inputs)


# def hard_swish():
#     @tf.function
#     def _func(inputs: tf.Tensor):
#         return inputs * hard_sigmoid(inputs)

#     return _func  # K.hard_sigmoid(inputs) * inputs


def squeeze_and_excite(inputs: tf.Tensor, ratio: int, name: str):
    filter_size = inputs.shape[-1]
    _x = Sequential(name=name,
                    layers=[
                        GlobalAveragePooling2D(),
                        Dense(filter_size // ratio, activation="relu"),
                        Dense(filter_size, activation=hard_sigmoid),
                    ])(inputs)
    _x = Multiply()([inputs, _x])

    return _x


# _activation = "relu"
_activation = HardSwish


def channel_shuffle(inputs: tf.Tensor, groups: int, name: str):
    _, height, width, in_channels = inputs.shape
    channels_per_group = in_channels // groups

    _x = Sequential(name=name,
                    layers=[
                        Reshape((-1, groups, channels_per_group)),
                        Permute((1, 3, 2)),
                        Reshape((height, width, in_channels)),
                    ])(inputs)

    return _x


def _shuffleNetV2_block(inputs: tf.Tensor, output_channels: int, strides: int,
                        rate: int, name: str, downsample: bool):
    branch_features = output_channels // 2

    _x1, _x2 = tf.split(inputs, num_or_size_splits=2, axis=-1)

    if downsample:
        _x1 = Sequential(
            name=f"{name}_branch1",
            layers=[
                DepthwiseConv2D(kernel_size=3,
                                strides=strides,
                                padding="same",
                                dilation_rate=rate,
                                use_bias=False),
                BatchNormalization(),
                Conv2D(
                    branch_features,
                    kernel_size=1,
                    strides=1,
                    # activation=_activation,
                    padding="valid",
                    use_bias=False),
                BatchNormalization(),
                _activation(),
            ])(_x1)

    _x2 = Sequential(
        name=f"{name}_branch2",
        layers=[
            Conv2D(
                branch_features,
                kernel_size=1,
                strides=1,
                # activation=_activation,
                padding="valid",
                use_bias=False),
            BatchNormalization(),
            _activation(),
            DepthwiseConv2D(kernel_size=3,
                            strides=strides,
                            padding="same",
                            dilation_rate=rate,
                            use_bias=False),
            BatchNormalization(),
            Conv2D(
                branch_features,
                kernel_size=1,
                strides=1,
                # activation=_activation,
                padding="valid",
                use_bias=False),
            BatchNormalization(),
            _activation(),
        ])(_x2)
    if not downsample:
        _x2 = squeeze_and_excite(_x2, 4, name=f"{name}_branch2_se")

    _out = Concatenate()([_x1, _x2])
    _out = channel_shuffle(_out, 2, name=f"{name}_shuffle")

    return _out


def shufflenet_v2_base(inputs: tf.Tensor,
                       stages_repeats: list = [2, 4, 2],
                       stages_out_channels: list = [24, 116, 232, 464, 1024],
                       output_stride: int = 32,
                       prefix: str = "shufflenet_v2"):
    if len(stages_repeats) != 3:
        raise ValueError('expected stages_repeats as list of 3 positive ints')
    if len(stages_out_channels) != 5:
        raise ValueError(
            'expected stages_out_channels as list of 5 positive ints')
    if output_stride < 4:
        raise ValueError('output stride cannot be smaller than 4')

    current_stride = 1
    current_rate = 1
    branch_exits = {}

    output_channels = stages_out_channels[0]
    _x = Sequential(
        name=f"{prefix}_conv1",
        layers=[
            Conv2D(
                output_channels,
                kernel_size=3,
                strides=2,
                # activation=_activation,
                padding="same",
                use_bias=False),
            BatchNormalization(),
            _activation(),
        ])(inputs)
    current_stride *= 2
    branch_exits[str(current_stride)] = _x

    _x = layers.MaxPooling2D(pool_size=3,
                             strides=2,
                             padding="same",
                             name=f"{prefix}_maxpool2d")(_x)
    current_stride *= 2
    branch_exits[str(current_stride)] = _x

    stage_names = [f'{prefix}_stage{i}' for i in [2, 3, 4]]
    for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                              stages_out_channels[1:]):
        layer_stride = 2
        if current_stride == output_stride:
            rate_multiplier = layer_stride
            layer_stride = 1

        _x = _shuffleNetV2_block(_x,
                                 output_channels,
                                 strides=layer_stride,
                                 rate=current_rate,
                                 downsample=True,
                                 name=f"{name}_downsample")

        if current_stride == output_stride:
            current_rate *= rate_multiplier
        current_stride *= layer_stride

        for i in range(repeats - 1):
            _x = _shuffleNetV2_block(_x,
                                     output_channels,
                                     strides=1,
                                     rate=current_rate,
                                     downsample=False,
                                     name=f"{name}_basic{i}")

        if layer_stride == 2:
            branch_exits[str(current_stride)] = _x

    return _x, branch_exits


def shufflenet_v2(inputs: tf.Tensor,
                  num_classes: int = 1000,
                  stages_repeats: list = [2, 4, 2],
                  stages_out_channels: list = [24, 116, 232, 464, 1024],
                  output_stride: int = 32):
    prefix = "shufflenet_v2"

    _x, _ = shufflenet_v2_base(inputs,
                               stages_repeats=stages_repeats,
                               stages_out_channels=stages_out_channels,
                               output_stride=output_stride,
                               prefix=prefix)

    output_channels = stages_out_channels[-1]
    _x = Sequential(
        name=f"{prefix}_conv_last",
        layers=[
            Conv2D(
                output_channels,
                kernel_size=1,
                strides=1,
                # activation=_activation,
                padding="valid",
                use_bias=False),
            BatchNormalization(),
            _activation(),
        ])(_x)

    _x = layers.GlobalAveragePooling2D(name=f"{prefix}_global_avg_pool")(_x)
    _x = layers.Dense(num_classes,
                      name=f"{prefix}_logits",
                      activation="softmax")(_x)

    return _x


if __name__ == "__main__":
    inputs = keras.Input(shape=(224, 224, 3))
    outputs = shufflenet_v2(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('shufflenet_v2.tflite', 'wb') as f:
        f.write(tflite_model)
