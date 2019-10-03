"""
Lovasz-Softmax and Jaccard hinge loss in Tensorflow
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division

import tensorflow as tf


class SoftmaxCrossEntropy(tf.losses.Loss):
    def __init__(self, num_classes: int, ignore_label: int, *args, **kwargs):
        super(SoftmaxCrossEntropy, self).__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.ignore_label = ignore_label

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor, *args, **kwargs):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.nn.softmax(y_pred)

        probas = tf.reshape(y_pred, [-1, self.num_classes])
        labels = tf.reshape(y_true, [-1, ])

        valid = tf.not_equal(labels, self.ignore_label)

        vprobas = tf.boolean_mask(tensor=probas, mask=valid)
        vlabels = tf.boolean_mask(tensor=labels, mask=valid)
        one_hot_labels = tf.one_hot(vlabels, self.num_classes)

        return tf.keras.losses.categorical_crossentropy(one_hot_labels, vprobas)


class LovaszSoftmax(tf.losses.Loss):
    def __init__(self, num_classes: int, ignore_label: int, *args, **kwargs):
        super(LovaszSoftmax, self).__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.ignore_label = ignore_label

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor, *args, **kwargs):
        y_pred = tf.nn.softmax(y_pred)

        loss = lovasz_softmax(y_pred,
                              y_true,
                              classes='present',
                              per_image=False,
                              ignore=self.ignore_label)

        return loss


@tf.function
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(input_tensor=gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------


@tf.function
def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:

        @tf.function
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)

        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(input_tensor=losses)
    else:
        loss = lovasz_hinge_flat(
            *flatten_binary_scores(logits, labels, ignore))
    return loss


@tf.function
def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    @tf.function
    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors,
                                          k=tf.shape(input=errors)[0],
                                          name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted),
                            tf.stop_gradient(grad),
                            1,
                            name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(pred=tf.equal(tf.shape(input=logits)[0], 0),
                   true_fn=lambda: tf.reduce_sum(input_tensor=logits) * 0.,
                   false_fn=compute_loss,
                   name="loss")
    return loss


@tf.function
def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1, ))
    labels = tf.reshape(labels, (-1, ))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(tensor=scores, mask=valid, name='valid_scores')
    vlabels = tf.boolean_mask(tensor=labels, mask=valid, name='valid_labels')
    return vscores, vlabels


# --------------------------- MULTICLASS LOSSES ---------------------------


@tf.function
def lovasz_softmax(probas,
                   labels,
                   classes='present',
                   per_image=False,
                   ignore=None,
                   order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:

        @tf.function
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)

        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(input_tensor=losses)
    else:
        loss = lovasz_softmax_flat(
            *flatten_probas(probas, labels, ignore, order), classes=classes)
    return loss


@tf.function
def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = probas.shape[1]
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = tf.cast(tf.equal(labels, c),
                     probas.dtype)  # foreground for class c
        if classes == 'present':
            present.append(tf.reduce_sum(input_tensor=fg) > 0)
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = tf.abs(fg - class_pred)
        errors_sorted, perm = tf.nn.top_k(errors,
                                          k=tf.shape(input=errors)[0],
                                          name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted,
                         tf.stop_gradient(grad),
                         1,
                         name="loss_class_{}".format(c)))
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(tensor=losses_tensor, mask=present)
    loss = tf.reduce_mean(input_tensor=losses_tensor)
    return loss


@tf.function
def flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if len(probas.shape) == 3:
        probas, order = tf.expand_dims(probas, 3), 'BHWC'
    if order == 'BCHW':
        probas = tf.transpose(a=probas, perm=(0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1, ))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(tensor=probas, mask=valid, name='valid_probas')
    vlabels = tf.boolean_mask(tensor=labels, mask=valid, name='valid_labels')
    return vprobas, vlabels
