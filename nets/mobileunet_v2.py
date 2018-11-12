from keras import layers, backend
from keras_applications.mobilenet_v2 import _make_divisible, correct_pad

BASE_WEIGHT_PATH = ('https://github.com/JonathanCMitchell/mobilenet_v2_keras/'
                    'releases/download/v1.1/')


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = backend.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3),
                                 name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def MobileUNet_v2(
        input_tensor,
        alpha=1.0,
        classes=3):
    # Alias
    img_input = input_tensor

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.ZeroPadding2D(padding=correct_pad(backend, img_input, 3),
                             name='Conv1_pad')(img_input)
    x = layers.Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name='Conv1')(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0)
    x1 = x
    print('x1: {}'.format(x1.shape))

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2)
    x2 = x
    print('x2: {}'.format(x2.shape))

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5)
    x3 = x
    print('x3: {}'.format(x3.shape))

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12)
    x4 = x
    print('x4: {}'.format(x4.shape))

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=14)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=15)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                            expansion=6, block_id=16)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(last_block_filters,
                      kernel_size=1,
                      use_bias=False,
                      name='Conv_1')(x)
    x = layers.BatchNormalization(epsilon=1e-3,
                                  momentum=0.999,
                                  name='Conv_1_bn')(x)
    x = layers.ReLU(6., name='out_relu')(x)

    x5 = x
    print('x5: {}'.format(x5.shape))

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1000, activation='softmax',
                     use_bias=True, name='Logits')(x)

    up1 = layers.concatenate(
        [
            x4,
            layers.Conv2DTranspose(
                96, (2, 2), strides=(2, 2), padding='same')(x5)
        ],
        axis=3
    )
    up1 = _inverted_res_block(up1, filters=96, alpha=alpha, stride=1,
                              expansion=6, block_id=17)
    print('up1: {}'.format(up1.shape))

    up2 = layers.concatenate(
        [
            x3,
            layers.Conv2DTranspose(
                32, (2, 2), strides=(2, 2), padding='same')(up1)
        ],
        axis=3
    )
    up2 = _inverted_res_block(up2, filters=32, alpha=alpha, stride=1,
                              expansion=6, block_id=18)
    print('up2: {}'.format(up2.shape))

    up3 = layers.concatenate(
        [
            x2,
            layers.Conv2DTranspose(
                24, (2, 2), strides=(2, 2), padding='same')(up2)
        ],
        axis=3
    )
    up3 = _inverted_res_block(up3, filters=24, alpha=alpha, stride=1,
                              expansion=6, block_id=19)
    print('up3: {}'.format(up3.shape))

    up4 = layers.concatenate(
        [
            x1,
            layers.Conv2DTranspose(
                16, (2, 2), strides=(2, 2), padding='same')(up3)
        ],
        axis=3
    )
    up4 = _inverted_res_block(up4, filters=16, alpha=alpha, stride=1,
                              expansion=6, block_id=20)
    print('up4: {}'.format(up4.shape))

    conv_last = layers.Conv2D(16, 1)(up4)
    print('conv_last: {}'.format(conv_last.shape))

    conv_score = layers.Conv2D(classes, 1)(conv_last)
    print('conv_score: {}'.format(conv_score.shape))

    out = layers.Activation('sigmoid', name='output_1')(conv_score)

    return out

def load_mobilenet_weights(model, alpha, dim):
    # Load weights.
    model_name = ('mobilenet_v2_weights_tf_dim_ordering_tf_kernels_' +
                  str(alpha) + '_' + str(dim) + '.h5')
    weigh_path = BASE_WEIGHT_PATH + model_name
    weights_path = keras.utils.get_file(
        model_name, weigh_path, cache_subdir='models')
    model.load_weights(weights_path, by_name=True)

if __name__ == "__main__":
    import keras
    import os

    input_tensor = layers.Input(shape=(224, 224, 3), name='input_1')
    net = MobileUNet_v2(
        input_tensor=input_tensor
    )

    # Create model.
    model = keras.models.Model(input_tensor, net)

    load_mobilenet_weights(model, 1.0, 224)

    model.summary()

    if not os.path.exists('./dist'):
        os.makedirs('./dist/')

    model.save('./dist/mobilenetv2.h5')
