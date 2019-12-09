import tensorflow as tf
from tensorflow import keras


class WeightedSparseMeanIoU(keras.metrics.MeanIoU):
    def __init__(self, num_classes: int, ignore_label: int = 255):
        super(WeightedSparseMeanIoU, self).__init__(num_classes=num_classes)
        self.ignore_label = ignore_label

    def update_state(self, y_true, y_pred, *args, **kwargs):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.argmax(y_pred, axis=-1)

        probas = tf.reshape(y_pred, [-1])
        labels = tf.reshape(y_true, [-1])

        valid = tf.not_equal(labels, self.ignore_label)

        vprobas = tf.boolean_mask(tensor=probas, mask=valid)
        vlabels = tf.boolean_mask(tensor=labels, mask=valid)

        super(WeightedSparseMeanIoU, self).update_state(vlabels, vprobas)


class WeightedAccuracy(keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, num_classes: int, ignore_label: int = 255):
        super(WeightedAccuracy, self).__init__()
        self.ignore_label = ignore_label
        self.num_classes = num_classes

    def update_state(self, y_true, y_pred, *args, **kwargs):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.nn.softmax(y_pred)

        probas = tf.reshape(y_pred, [-1, self.num_classes])
        labels = tf.reshape(y_true, [-1])

        valid = tf.not_equal(labels, self.ignore_label)

        vprobas = tf.boolean_mask(tensor=probas, mask=valid)
        vlabels = tf.boolean_mask(tensor=labels, mask=valid)
        # vlabels = tf.one_hot(vlabels, self.num_classes)

        super(WeightedAccuracy, self).update_state(vlabels, vprobas)

if __name__ == "__main__":
    a = tf.random.uniform([16, 225, 225, 2])
    b = tf.random.uniform([16, 225, 225], maxval=3, dtype=tf.int32)

    metric = WeightedAccuracy(2, 2)

    print(metric(b, a))
