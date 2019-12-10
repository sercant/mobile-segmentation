import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, regularizers, Sequential
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Concatenate, Activation, UpSampling2D
from core.shufflenet_v2 import shufflenet_v2_base
from core.encoder_heads import dpc_head, basic_head


def _batch_normalization():
    return layers.BatchNormalization(momentum=0.9997, epsilon=1e-5)


def l2_regulizer():
    return regularizers.l2(0.00004)


def encoder_heads(inputs: tf.Tensor,
                  use_dpc: bool = True,
                  filter_per_branch: int = 256):
    _x = dpc_head(
        inputs,
        filter_per_branch=filter_per_branch) if use_dpc else basic_head(
            inputs, filter_per_branch=filter_per_branch)
    return _x


def shufflenet_v2_segmentation(inputs: tf.Tensor,
                               number_of_classes: int,
                               output_stride: int = 16,
                               use_dpc: bool = True,
                               output_size: list = None,
                               feature_extractor_multiplier: float = 1.0,
                               filter_per_encoder_branch: int = 256,
                               decoder_stride: int = None,
                               feature_extractor_checkpoint: str = None,
                               small_backend: bool = False,
                               weight_loss: float = 0.00004):
    _x, branch_exits = shufflenet_v2_base(inputs,
                                          output_stride=output_stride,
                                          weight_loss=weight_loss)
    if feature_extractor_checkpoint is not None:
        tmp = keras.Model(inputs=inputs, outputs=_x)
        tmp.load_weights(feature_extractor_checkpoint, by_name=True)

    _x = encoder_heads(_x,
                       use_dpc,
                       filter_per_branch=filter_per_encoder_branch)

    if decoder_stride is not None:
        branch = branch_exits[str(decoder_stride)]

        branch = Sequential(name="decoder",
                            layers=[
                                Conv2D(48,
                                       kernel_size=1,
                                       strides=1,
                                       padding="same",
                                       use_bias=False,
                                       kernel_regularizer=l2_regulizer()),
                                _batch_normalization(),
                                Activation("relu")
                            ])(branch)

        _x = tf.image.resize(_x, branch.shape[1:3])
        # _x = tf.compat.v1.image.resize(_x, shape, align_corners=True)

        # scale = (branch.shape[1] // _x.shape[1],
        #          branch.shape[2] // _x.shape[2])
        # _x = UpSampling2D(scale, interpolation='bilinear')(_x)

        _x = Concatenate()([_x, branch])

        for i in range(2):
            _x = Sequential(name=f"decoder_conv_{i+1}",
                            layers=[
                                DepthwiseConv2D(kernel_size=3,
                                                strides=1,
                                                padding="same",
                                                use_bias=False),
                                _batch_normalization(),
                                Activation("relu"),
                                Conv2D(filter_per_encoder_branch,
                                       kernel_size=1,
                                       strides=1,
                                       padding="same",
                                       use_bias=False,
                                       kernel_regularizer=l2_regulizer()),
                                _batch_normalization(),
                                Activation("relu"),
                            ])(_x)

    # _x = layers.Dropout(0.1)(_x)

    _x = Conv2D(number_of_classes,
                kernel_size=1,
                strides=1,
                padding="same",
                name="logits",
                use_bias=True,
                kernel_regularizer=l2_regulizer())(_x)

    if output_size is not None and len(output_size) != 2:
        raise ValueError("Expected output size length of 2 but got {}.".format(
            len(output_size)))
    else:
        output_size = inputs.shape[1:3]

    _x = tf.image.resize(_x, output_size)
    # _x = tf.compat.v1.image.resize(_x, output_size, align_corners=True)

    # scale = (output_size[0] // _x.shape[1], output_size[1] // _x.shape[2])
    # _x = UpSampling2D(scale, interpolation='bilinear')(_x)

    return _x


if __name__ == "__main__":
    inputs = keras.Input(shape=[225, 225, 3])
    output = shufflenet_v2_segmentation(inputs,
                                        19,
                                        16,
                                        use_dpc=False,
                                        decoder_stride=8,
                                        filter_per_encoder_branch=256)

    model = keras.Model(inputs=inputs, outputs=output)

    model.summary()
    model.save("./checkpoints/shufflenet_v2_seg.h5")

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('./checkpoints/shufflenet_v2_seg.tflite', 'wb') as f:
        f.write(tflite_model)
