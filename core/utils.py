
"""This script contains utility functions."""
import tensorflow as tf
from tensorflow.contrib import slim


def scale_dimension(dim, scale):
    """Scales the input dimension.

    Args:
      dim: Input dimension (a scalar or a scalar Tensor).
      scale: The amount of scaling applied to the input.

    Returns:
      Scaled dimension.
    """
    if isinstance(dim, tf.Tensor):
        return tf.cast((tf.cast(dim, tf.float32) - 1.0) * scale + 1.0, dtype=tf.int32)
    else:
        return int((float(dim) - 1.0) * scale + 1.0)


def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None):
    """Splits a separable conv2d into depthwise and pointwise conv2d.

    This operation differs from `tf.layers.separable_conv2d` as this operation
    applies activation function between depthwise and pointwise conv2d.

    Args:
      inputs: Input tensor with shape [batch, height, width, channels].
      filters: Number of filters in the 1x1 pointwise convolution.
      kernel_size: A list of length 2: [kernel_height, kernel_width] of
        of the filters. Can be an int if both values are the same.
      rate: Atrous convolution rate for the depthwise convolution.
      weight_decay: The weight decay to use for regularizing the model.
      depthwise_weights_initializer_stddev: The standard deviation of the
        truncated normal weight initializer for depthwise convolution.
      pointwise_weights_initializer_stddev: The standard deviation of the
        truncated normal weight initializer for pointwise convolution.
      scope: Optional scope for the operation.

    Returns:
      Computed features after split separable conv2d.
    """
    outputs = slim.separable_conv2d(
        inputs,
        None,
        kernel_size=kernel_size,
        depth_multiplier=1,
        rate=rate,
        weights_initializer=tf.truncated_normal_initializer(
            stddev=depthwise_weights_initializer_stddev),
        weights_regularizer=None,
        scope=scope + '_depthwise')
    return slim.conv2d(
        outputs,
        filters,
        1,
        weights_initializer=tf.truncated_normal_initializer(
            stddev=pointwise_weights_initializer_stddev),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        scope=scope + '_pointwise')
