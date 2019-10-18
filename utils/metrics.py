import tensorflow as tf
from tensorflow import keras


class WeightedSparseMeanIoU(keras.metrics.MeanIoU):
    def __init__(self, num_classes: int, ignore_label: int = 255):
        super(WeightedSparseMeanIoU, self).__init__(num_classes=num_classes)
        self.ignore_label = ignore_label

    def update_state(self, y_true, y_pred, sample_weight=None):
        _ke = keras.backend
        ignored_y_true = tf.where(_ke.equal(y_true, self.ignore_label),
                                  _ke.zeros_like(y_true), y_true)

        y_pred = _ke.argmax(y_pred)

        sample_weight = _ke.cast(
            _ke.squeeze(_ke.not_equal(y_true, self.ignore_label), axis=-1),
            'float32')

        super(WeightedSparseMeanIoU,
              self).update_state(ignored_y_true, y_pred, sample_weight)


class WeightedAccuracy(keras.metrics.Accuracy):
    def __init__(self, ignore_label: int = 255):
        super(WeightedAccuracy, self).__init__()
        self.ignore_label = ignore_label

    def update_state(self, y_true, y_pred, sample_weight=None):
        _ke = keras.backend
        ignored_y_true = tf.where(_ke.equal(y_true, self.ignore_label),
                                  _ke.zeros_like(y_true), y_true)

        y_pred = _ke.argmax(y_pred)

        sample_weight = _ke.cast(
            _ke.squeeze(_ke.not_equal(y_true, self.ignore_label), axis=-1),
            'float32')

        super(WeightedAccuracy, self).update_state(ignored_y_true, y_pred,
                                                   sample_weight)
