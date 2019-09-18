import tensorflow as tf


class IgnoreLabeledMeanIoU(tf.metrics.MeanIoU):
    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 name=None,
                 dtype=None):
        super(IgnoreLabeledMeanIoU, self).__init__(num_classes=num_classes,
                                                   name=name,
                                                   dtype=dtype)
        self.ignore_label = ignore_label

    @tf.function
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                     sample_weight):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.argmax(y_pred, axis=-1)

        _predictions = tf.reshape(y_pred, shape=[-1])
        _labels = tf.reshape(y_true, shape=[-1])
        _weights = tf.cast(tf.not_equal(_labels, self.ignore_label),
                           tf.float32)
        _labels = tf.where(tf.equal(_labels, self.ignore_label),
                           tf.zeros_like(_labels), _labels)

        return super(IgnoreLabeledMeanIoU,
                     self).update_state(_labels, _predictions, _weights)
