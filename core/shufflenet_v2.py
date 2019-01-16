import tensorflow as tf

slim = tf.contrib.slim

BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-3


def training_scope(is_training=True,
                   weight_decay=0.00004,
                   stddev=0.09,
                   dropout_keep_prob=0.8,
                   bn_decay=0.997):
    with slim.arg_scope([shufflenet, shufflenet_base], is_training=is_training) as s:
        return s


@slim.add_arg_scope
def shufflenet_base(input_tensor,
                    depth_multiplier=1.0,
                    scope='ShuffleNetV2',
                    is_training=False,
                    output_stride=None,
                    **kwargs):
    possibilities = {0.5: 48, 1.0: 116, 1.5: 176, 2.0: 224}
    initial_depth = possibilities[depth_multiplier]
    end_points = {}

    if output_stride is not None:
        if output_stride == 0 or output_stride < 4 or (output_stride > 1 and output_stride % 2):
            raise ValueError(
                'Output stride must be None, or >= 4 and a multiple of 2.')
    current_stride = 1
    layer_stride = 2

    def get_stride(layer_stride, current_stride):
        if current_stride == output_stride:
            return 1, current_stride
        else:
            current_stride *= layer_stride
            return layer_stride, current_stride

    def batch_norm(net):
        net = tf.layers.batch_normalization(
            net, axis=3, center=True, scale=True,
            training=is_training,
            momentum=BATCH_NORM_MOMENTUM,
            epsilon=BATCH_NORM_EPSILON,
            fused=True, name='batch_norm'
        )
        return net

    with tf.variable_scope(scope):
        params = {
            'padding': 'SAME', 'activation_fn': tf.nn.relu,
            'normalizer_fn': batch_norm, 'data_format': 'NHWC',
            'weights_initializer': tf.contrib.layers.xavier_initializer()
        }
        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):
            layer_stride, current_stride = get_stride(2, current_stride)
            net = slim.conv2d(input_tensor, 24, (3, 3),
                              stride=layer_stride, scope='Conv1')
            layer_stride, current_stride = get_stride(2, current_stride)
            net = slim.max_pool2d(net, (3, 3), stride=layer_stride,
                                  padding='SAME', scope='MaxPool')
            end_points['Stage1'] = net

            layers = [
                {'num_units': 4, 'out_channels': initial_depth,
                    'scope': 'Stage2', 'stride': 2},
                {'num_units': 8, 'out_channels': None,
                    'scope': 'Stage3', 'stride': 2},
                {'num_units': 4, 'out_channels': None,
                    'scope': 'Stage4', 'stride': 2},
            ]

            for i in range(3):
                layer = layers[i]
                layer_stride, current_stride = get_stride(
                    layer['stride'], current_stride)
                net = block(
                    net,
                    num_units=layer['num_units'],
                    out_channels=layer['out_channels'],
                    scope=layer['scope'],
                    stride=layer_stride)

    return net, end_points


@slim.add_arg_scope
def shufflenet(input_tensor,
               num_classes=1000,
               depth_multiplier=1.0,
               scope='ShuffleNetV2',
               base_only=False,
               is_training=False,
               **kwargs):
    """
    This is an implementation of ShuffleNet v2:
    https://arxiv.org/abs/1807.11164

    Arguments:
        images: a float tensor with shape [batch_size, image_height, image_width, 3],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        num_classes: an integer.
        depth_multiplier: a float, possible values are 0.5, 1.0, 1.5, and 2.0.
    Returns:
        a float tensor with shape [batch_size, num_classes].
    """
    net, end_points = shufflenet_base(input_tensor,
                                      depth_multiplier=depth_multiplier,
                                      scope=scope,
                                      is_training=is_training,
                                      **kwargs)
    if base_only:
        return net, end_points

    final_channels = 1024 if depth_multiplier != '2.0' else 2048
    net = slim.conv2d(net, final_channels, (1, 1),
                      stride=1, scope='Conv5')
    end_points['final_channels'] = net

    # global average pooling
    net = tf.reduce_mean(net, axis=[1, 2])

    logits = slim.fully_connected(
        net, num_classes, activation_fn=None, scope='classifier',
        weights_initializer=tf.contrib.layers.xavier_initializer()
    )
    end_points['logits'] = logits

    return logits, end_points


def block(x, num_units, out_channels=None, scope='stage', stride=2):
    with tf.variable_scope(scope):

        with tf.variable_scope('unit_1'):
            x, y = basic_unit_with_downsampling(x, out_channels, stride=stride)

        for j in range(2, num_units + 1):
            with tf.variable_scope('unit_%d' % j):
                x, y = concat_shuffle_split(x, y)
                x = basic_unit(x)
        x = tf.concat([x, y], axis=3)

    return x


def concat_shuffle_split(x, y):
    with tf.name_scope('concat_shuffle_split'):
        shape = tf.shape(x)
        batch_size = shape[0]
        height, width = shape[1], shape[2]
        depth = x.shape[3].value

        # shape [batch_size, height, width, 2, depth]
        z = tf.stack([x, y], axis=3)
        z = tf.transpose(z, [0, 1, 2, 4, 3])
        z = tf.reshape(z, [batch_size, height, width, 2*depth])
        x, y = tf.split(z, num_or_size_splits=2, axis=3)
        return x, y


def basic_unit(x):
    in_channels = x.shape[3].value
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    x = depthwise_conv(x, kernel=3, stride=1,
                       activation_fn=None, scope='depthwise')
    x = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_after')
    return x


def basic_unit_with_downsampling(x, out_channels=None, stride=2):
    in_channels = x.shape[3].value
    out_channels = 2 * in_channels if out_channels is None else out_channels

    y = slim.conv2d(x, in_channels, (1, 1), stride=1, scope='conv1x1_before')
    y = depthwise_conv(y, kernel=3, stride=stride,
                       activation_fn=None, scope='depthwise')
    y = slim.conv2d(y, out_channels // 2, (1, 1),
                    stride=1, scope='conv1x1_after')

    with tf.variable_scope('second_branch'):
        x = depthwise_conv(x, kernel=3, stride=stride,
                           activation_fn=None, scope='depthwise')
        x = slim.conv2d(x, out_channels // 2, (1, 1),
                        stride=1, scope='conv1x1_after')
        return x, y


@tf.contrib.framework.add_arg_scope
def depthwise_conv(
        x, kernel=3, stride=1, padding='SAME',
        activation_fn=None, normalizer_fn=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        data_format='NHWC', scope='depthwise_conv'):

    with tf.variable_scope(scope):
        assert data_format == 'NHWC'
        in_channels = x.shape[3].value
        W = tf.get_variable(
            'depthwise_weights',
            [kernel, kernel, in_channels, 1], dtype=tf.float32,
            initializer=weights_initializer
        )
        x = tf.nn.depthwise_conv2d(
            x, W, [1, stride, stride, 1], padding, data_format='NHWC')
        # batch normalization
        x = normalizer_fn(x) if normalizer_fn is not None else x
        x = activation_fn(
            x) if activation_fn is not None else x  # nonlinearity
        return x
