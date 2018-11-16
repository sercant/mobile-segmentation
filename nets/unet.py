from keras import layers, utils

def conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1), block_id=1):
    channel_axis = -1
    filters = int(filters * alpha)
    x = layers.Conv2D(filters, kernel,
                      padding='same',
                      use_bias=False,
                      strides=strides,
                      name='conv%d' % block_id)(inputs)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_%d_relu' % block_id)(x)

def depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    # if strides == (1, 1):
    #     x = inputs
    # else:
    #     x = layers.ZeroPadding2D(((0, 1), (0, 1)),
    #                              name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(inputs)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

def conv_transpose_block(inputs, filters, kernel_size=2):
    net = layers.Conv2DTranspose(filters, kernel_size, strides=2, padding='same')(inputs)
    net = layers.BatchNormalization()(net)
    net = layers.ReLU(6.)(net)

    return net


def unet(inputs, num_classes):
    alpha = 1.0
    depth_multiplier = 1

    x = conv_block(inputs, 32, alpha, strides=(2, 2), block_id=1)
    skip_0 = x

    x = depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    # x = layers.MaxPooling2D(strides=2) (x)
    skip_1 = x

    x = depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2)
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    # x = layers.MaxPooling2D(strides=2) (x)
    skip_2 = x

    x = depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    # x = layers.MaxPooling2D(strides=2) (x)
    skip_3 = x

    x = depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    # x = layers.MaxPooling2D(strides=2) (x)
    skip_4 = x

    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)
    # x = layers.MaxPooling2D(strides=2) (x)

    x = layers.concatenate([
        conv_transpose_block(x, 512),
        skip_4,
    ], axis=3)
    x = depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=14)

    x = layers.concatenate([
        conv_transpose_block(x, 256),
        skip_3,
    ], axis=3)
    x = depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=15)

    x = layers.concatenate([
        conv_transpose_block(x, 128),
        skip_2,
    ], axis=3)
    x = depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=16)

    x = layers.concatenate([
        conv_transpose_block(x, 64),
        skip_1,
    ], axis=3)
    x = depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=17)

    x = layers.concatenate([x, skip_0], axis=3)
    # x = depthwise_conv_block(x, 32, alpha_up, depth_multiplier, block_id=18)
    x = conv_block(x, 32, alpha, block_id=18)

    x = layers.Conv2D(
        num_classes, 1, activation='sigmoid', name='output_1')(x)

    return x

def load_unet_weights(model, image_size):
    # Load weights.
    model_name = 'mobilenet_1_0_%d_tf.h5' % image_size
    weigh_path = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_%d_tf.h5' % image_size
    weights_path = utils.get_file(
        model_name, weigh_path, cache_subdir='models')
    model.load_weights(weights_path, by_name=True)
