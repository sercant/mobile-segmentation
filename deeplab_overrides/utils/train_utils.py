import six

import tensorflow as tf

from deeplab.core import preprocess_utils
from deeplab.utils import train_utils as _super

slim = _super.slim


def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    # old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    # new haodong
    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice

def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  num_classes,
                                                  ignore_label,
                                                  loss_weights,
                                                  upsample_logits=True,
                                                  scope=None):
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
            masks.append(tf.to_float(tf.equal(scaled_labels, i)) * loss_weights[i])
        not_ignore_mask = sum(masks)
        # not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
        #                                            ignore_label)) * loss_weight
        one_hot_labels = slim.one_hot_encoding(
            scaled_labels, num_classes, on_value=1.0, off_value=0.0)
        flattened_output = tf.reshape(logits, shape=[-1, num_classes])
        tf.losses.softmax_cross_entropy(
            one_hot_labels,
            flattened_output,
            weights=not_ignore_mask,
            scope=loss_scope)

        # from tensorflow.python.ops.losses import util
        # util.add_loss(1.0 - dice_coe(flattened_output,
        #                              one_hot_labels, axis=[1]))


get_model_init_fn = _super.get_model_init_fn

get_model_gradient_multipliers = _super.get_model_gradient_multipliers

get_model_learning_rate = _super.get_model_learning_rate
