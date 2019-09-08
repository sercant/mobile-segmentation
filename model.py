import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from shufflenet_v2 import shufflenet_v2_base
from encoder_heads import dpc_head, basic_head

# This is to fix the bug of https://github.com/tensorflow/tensorflow/issues/27298
# if it gets fixed remove the related lines
from tensorflow.python.keras.backend import get_graph


def encoder_heads(inputs: tf.Tensor,
                  use_dpc: bool = True,
                  weight_decay: float = 0.00004):
    _x = dpc_head(inputs,
                  weight_decay=weight_decay) if use_dpc else basic_head(
                      inputs, weight_decay=weight_decay)
    return _x


def shufflenet_v2_segmentation(inputs: tf.Tensor,
                               number_of_classes: int,
                               output_stride: int = 16,
                               use_dpc: bool = True,
                               output_size: list = None,
                               feature_extractor_multiplier: float = 1.0,
                               weight_decay: float = 0.00004):
    _x = shufflenet_v2_base(inputs,
                            feature_extractor_multiplier,
                            output_stride,
                            weight_decay=weight_decay)
    _x = encoder_heads(_x, use_dpc, weight_decay=weight_decay)

    with get_graph().as_default(), tf.name_scope("logits"):
        _x = layers.Conv2D(
            number_of_classes,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(weight_decay))(_x)

        _x = layers.Dropout(0.1)(_x)

        if output_size is not None and len(output_size) != 2:
            raise ValueError(
                "Expected output size length of 2 but got {}.".format(
                    len(output_size)))
        else:
            output_size = inputs.shape[1:3]

        _x = tf.image.resize(_x, output_size)

    return _x


if __name__ == "__main__":
    inputs = keras.Input(shape=(225, 225, 3))
    output = shufflenet_v2_segmentation(inputs, 19, 16, False)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    import h5py
    from utils import load_weights_from_hdf5_group_by_name

    with h5py.File("./checkpoints/cifar10.hdf5", 'r') as f:
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']
        load_weights_from_hdf5_group_by_name(f, model.layers, skip_mismatch=True)

    model.summary()
    model.save("shufflenet_v2.h5")
