import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from core.shufflenet_v2 import shufflenet_v2_base
from core.encoder_heads import dpc_head, basic_head

BATCH_NORM_PARAMS = {'decay': 0.9997, 'epsilon': 1e-5}


def batch_norm(inputs: tf.Tensor):
    _x = layers.BatchNormalization(momentum=BATCH_NORM_PARAMS['decay'],
                                   epsilon=BATCH_NORM_PARAMS['epsilon'],
                                   fused=True)(inputs)

    return _x


def encoder_heads(inputs: tf.Tensor,
                  use_dpc: bool = True,
                  weight_decay: float = 0.00004,
                  filter_per_branch: int = 256):
    _x = dpc_head(
        inputs, weight_decay=weight_decay,
        filter_per_branch=filter_per_branch) if use_dpc else basic_head(
            inputs,
            weight_decay=weight_decay,
            filter_per_branch=filter_per_branch)
    return _x


def shufflenet_v2_segmentation(inputs: tf.Tensor,
                               number_of_classes: int,
                               output_stride: int = 16,
                               use_dpc: bool = True,
                               output_size: list = None,
                               feature_extractor_multiplier: float = 1.0,
                               weight_decay: float = 0.00004,
                               filter_per_encoder_branch: int = 256,
                               decoder_stride: int = None,
                               feature_extractor_checkpoint: str = None,
                               small_backend: bool = False):
    _x, branch_exits = shufflenet_v2_base(inputs, output_stride=output_stride)
    # feature_extractor_multiplier,
    #   weight_decay=weight_decay,
    #   small_backend=small_backend)
    if feature_extractor_checkpoint is not None:
        tmp = keras.Model(inputs=inputs, outputs=_x)
        tmp.load_weights(feature_extractor_checkpoint, by_name=True)

    _x = encoder_heads(_x,
                       use_dpc,
                       weight_decay=weight_decay,
                       filter_per_branch=filter_per_encoder_branch)

    if decoder_stride is not None:
        with tf.name_scope("decoder"):
            branch = branch_exits[str(decoder_stride)]
            branch = layers.Conv2D(
                48,
                kernel_size=1,
                strides=1,
                activation="relu",
                padding="same",
                kernel_regularizer=keras.regularizers.l2(weight_decay))(branch)
            branch = batch_norm(branch)

            shape = tf.add(_x.shape[1:3], -1)
            shape = tf.multiply(shape, output_stride // decoder_stride)
            shape = tf.add(shape, 1)
            _x = tf.image.resize(_x, shape)

            _x = layers.Concatenate()([_x, branch])

            for i in range(2):
                _x = layers.DepthwiseConv2D(kernel_size=3,
                                            strides=1,
                                            activation="relu",
                                            padding="same")(_x)
                _x = batch_norm(_x)
                _x = layers.Conv2D(
                    filter_per_encoder_branch,
                    kernel_size=1,
                    strides=1,
                    padding="same",
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(weight_decay))(_x)
                _x = batch_norm(_x)

    _x = layers.Conv2D(number_of_classes,
                       kernel_size=1,
                       strides=1,
                       padding="same",
                       kernel_regularizer=keras.regularizers.l2(weight_decay),
                       name="logits")(_x)

    # _x = layers.Dropout(0.1)(_x)

    if output_size is not None and len(output_size) != 2:
        raise ValueError("Expected output size length of 2 but got {}.".format(
            len(output_size)))
    else:
        output_size = inputs.shape[1:3]

    _x = tf.image.resize(_x, output_size)

    return _x


if __name__ == "__main__":
    inputs = keras.Input(shape=[513, 513, 3])
    output = shufflenet_v2_segmentation(inputs,
                                        19,
                                        16,
                                        use_dpc=False,
                                        decoder_stride=None,
                                        filter_per_encoder_branch=256)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.summary()
    model.save("./checkpoints/shufflenet_v2_seg.h5")

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('./checkpoints/shufflenet_v2_seg.tflite', 'wb') as f:
        f.write(tflite_model)
