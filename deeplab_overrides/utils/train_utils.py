import six

import tensorflow as tf

from deeplab.core import preprocess_utils
from deeplab.utils import train_utils as _super

from metrics import dice_coefficient

slim = _super.slim


def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  num_classes,
                                                  ignore_label,
                                                  loss_weights=None,
                                                  upsample_logits=True,
                                                  scope=None,
                                                  add_jaccard_coef=False):
    """Adds softmax cross entropy loss for logits of each scale.

    Args:
      scales_to_logits: A map from logits names for different scales to logits.
        The logits have shape [batch, logits_height, logits_width, num_classes].
      labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
      num_classes: Integer, number of target classes.
      ignore_label: Integer, label to ignore.
      loss_weight: Float, loss weight.
      upsample_logits: Boolean, upsample logits or not.
      scope: String, the scope for the loss.

    Raises:
      ValueError: Label or logits is None.
    """
    if labels is None:
        raise ValueError('No label for softmax cross entropy loss.')

    if not loss_weights:
        loss_weights = [1.0 for i in range(num_classes)]

    for scale, logits in six.iteritems(scales_to_logits):
        loss_scope = None
        if scope:
            loss_scope = '%s_%s' % (scope, scale)

        if upsample_logits:
            # Label is not downsampled, and instead we upsample logits.
            logits = tf.image.resize_bilinear(
                logits,
                preprocess_utils.resolve_shape(labels, 4)[1:3],
                align_corners=True)
            scaled_labels = labels
        else:
            # Label is downsampled to the same size as logits.
            scaled_labels = tf.image.resize_nearest_neighbor(
                labels,
                preprocess_utils.resolve_shape(logits, 4)[1:3],
                align_corners=True)

        scaled_labels = tf.reshape(scaled_labels, shape=[-1])
        masks = []
        masks.append(tf.to_float(tf.equal(scaled_labels, ignore_label)) * 0.0)
        for i in range(num_classes):
            masks.append(tf.to_float(tf.equal(scaled_labels, i))
                         * loss_weights[i])
        not_ignore_mask = sum(masks)
        # not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
        #                                            ignore_label)) * loss_weight
        one_hot_labels = slim.one_hot_encoding(
            scaled_labels, num_classes, on_value=1.0, off_value=0.0)

        with tf.name_scope('pixel_wise_softmax'):
            softmax_logits = tf.nn.softmax(logits)

        flattened_output = tf.reshape(softmax_logits, shape=[-1, num_classes])

        tf.losses.add_loss(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=one_hot_labels, logits=flattened_output), axis=-1) + 1.0 - dice_coefficient(softmax_logits, one_hot_labels, smooth=1.)))
        # tf.losses.softmax_cross_entropy(
        #     one_hot_labels,
        #     flattened_output,
        #     weights=not_ignore_mask,
        #     scope=loss_scope)




get_model_init_fn = _super.get_model_init_fn

get_model_gradient_multipliers = _super.get_model_gradient_multipliers

get_model_learning_rate = _super.get_model_learning_rate
