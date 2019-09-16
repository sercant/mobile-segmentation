import tensorflow as tf
from tensorflow import keras

from dataset import get_dataset
from preprocess2 import preprocess
from model import shufflenet_v2_segmentation


class SoftmaxCrossEntropy(object):
    def __init__(self, num_classes: int, ignore_label: int):
        self.num_classes = num_classes
        self.ignore_label = ignore_label

    @tf.function
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, *args, **kwargs):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        # y_pred = tf.nn.softmax(y_pred)

        labels = tf.reshape(y_true, shape=[-1])
        not_ignore_mask = tf.cast(tf.not_equal(labels, self.ignore_label),
                                  tf.float32)
        one_hot_labels = tf.one_hot(labels, self.num_classes)

        y_pred = tf.reshape(y_pred, shape=[-1, self.num_classes])

        loss = tf.compat.v1.losses.softmax_cross_entropy(
            one_hot_labels, y_pred, weights=not_ignore_mask)
        return loss


class MIOU(tf.metrics.MeanIoU):
    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 name=None,
                 dtype=None):
        super(MIOU, self).__init__(num_classes=num_classes,
                                   name=name,
                                   dtype=dtype)
        self.ignore_label = ignore_label

    @tf.function
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.argmax(y_pred, axis=-1)

        _predictions = tf.reshape(y_pred, shape=[-1])
        _labels = tf.reshape(y_true, shape=[-1])
        _weights = tf.cast(tf.not_equal(_labels, self.ignore_label),
                           tf.float32)
        _labels = tf.where(tf.equal(_labels, self.ignore_label),
                           tf.zeros_like(_labels), _labels)

        return super(MIOU, self).update_state(_labels, _predictions, _weights)


dataset_name = "coco"
dataset_dir = "data/coco/tfrecord"
dataset_split = "val"
batch_size = 16
input_size = [225, 225]

dataset, dataset_desc = get_dataset(dataset_name, dataset_split, dataset_dir)
preprocessed_dataset = dataset.map(preprocess(dataset_desc,
                                              input_size)).batch(batch_size)


inputs = keras.Input(shape=(input_size[0], input_size[1], 3))
outputs = shufflenet_v2_segmentation(inputs, dataset_desc.num_classes)
model = keras.Model(inputs=inputs, outputs=outputs)


train_loss = SoftmaxCrossEntropy(dataset_desc.num_classes,
                                  dataset_desc.ignore_label)
train_accuracy = MIOU(dataset_desc.num_classes, dataset_desc.ignore_label)

optimizer = tf.keras.optimizers.Adam()


model.compile(loss=train_loss, optimizer=optimizer, metrics=[train_accuracy])
model.fit(preprocessed_dataset, epochs=3, steps_per_epoch=dataset_desc.splits_to_sizes[dataset_split] // batch_size)
