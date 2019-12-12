import tensorflow as tf
from tensorflow import keras

from dataset import get_dataset
from preprocess import preprocess
from model import shufflenet_v2_segmentation
from utils.losses import SoftmaxCrossEntropy, LovaszSoftmax
from utils.metrics import WeightedSparseMeanIoU, WeightedAccuracy

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

# Dataset related flags
flags.DEFINE_string('dataset', 'cityscapes', 'Name of the dataset.')
flags.DEFINE_string('dataset_split', 'val',
                    'Name of the dataset split for testing.')
flags.DEFINE_string('dataset_dir', None, 'Path of the dataset directory.')

# Training related flags
flags.DEFINE_integer('batch_size', 1, 'Batch size to use for training.')
flags.DEFINE_multi_integer('crop_size', [1025, 2049], 'Input crop size.')

flags.DEFINE_boolean('use_dpc', False, 'Use DPC architecture or not.')
flags.DEFINE_integer('decoder_stride', None, 'Decoder stride value.')

# Checkpoint related flags
flags.DEFINE_string('checkpoint_restore_path', None,
                    'Restore checkpoint path.')


def evaluate(_):
    dataset_name = FLAGS.dataset
    dataset_split = FLAGS.dataset_split
    dataset_dir = FLAGS.dataset_dir

    batch_size = FLAGS.batch_size
    input_size = FLAGS.crop_size

    use_dpc = FLAGS.use_dpc
    decoder_stride = FLAGS.decoder_stride

    checkpoint_restore_path = FLAGS.checkpoint_restore_path

    val_dataset, dataset_desc = get_dataset(dataset_name, dataset_split,
                                            dataset_dir)
    preprocessed_val_dataset = val_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE).map(
            preprocess(input_size,
                       is_training=False,
                       ignore_label=dataset_desc.ignore_label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)

    inputs = keras.Input(shape=(input_size[0], input_size[1], 3))
    outputs = shufflenet_v2_segmentation(inputs,
                                         dataset_desc.num_classes,
                                         filter_per_encoder_branch=256,
                                         use_dpc=use_dpc,
                                         decoder_stride=decoder_stride,
                                         output_stride=16)
    model = keras.Model(inputs=inputs, outputs=outputs)
    if checkpoint_restore_path is not None:
        model.load_weights(checkpoint_restore_path)

    train_loss = SoftmaxCrossEntropy(dataset_desc.num_classes,
                                     dataset_desc.ignore_label)
    miou = WeightedSparseMeanIoU(dataset_desc.num_classes,
                                 dataset_desc.ignore_label)
    accuracy = WeightedAccuracy(dataset_desc.num_classes,
                                dataset_desc.ignore_label)

    model.compile(loss=train_loss, metrics=[miou, accuracy])
    model.evaluate(preprocessed_val_dataset,
                   steps=dataset_desc.splits_to_sizes[dataset_split] //
                   batch_size)


if __name__ == "__main__":
    flags.mark_flag_as_required('dataset_dir')
    flags.mark_flag_as_required('checkpoint_restore_path')
    app.run(evaluate)
