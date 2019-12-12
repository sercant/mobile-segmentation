import tensorflow as tf
from tensorflow import keras

from dataset import get_dataset
from preprocess import preprocess
from model import shufflenet_v2_segmentation
from utils.losses import SoftmaxCrossEntropy
from utils.metrics import WeightedSparseMeanIoU, WeightedAccuracy

from absl import app
from absl import flags
from absl import logging

# import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

# Dataset related flags
flags.DEFINE_string('dataset', 'cityscapes', 'Name of the dataset.')
flags.DEFINE_string('dataset_train_split', 'train',
                    'Name of the dataset split for training.')
flags.DEFINE_string('dataset_test_split', 'val',
                    'Name of the dataset split for testing.')
flags.DEFINE_string('dataset_dir', None, 'Path of the dataset directory.')

# Training related flags
flags.DEFINE_integer('batch_size', 16, 'Batch size to use for training.')
flags.DEFINE_multi_integer('crop_size', [769, 769], 'Input crop size.')
flags.DEFINE_integer('num_epochs', 60, 'Train the network for num_epochs.')

flags.DEFINE_enum('optimizer', 'adam', ['adam', 'sgd'], 'Training optimizer.')
flags.DEFINE_float('initial_lr', 0.001, 'Initial learning rate.')
flags.DEFINE_enum('lr_decay_policy', 'plateau', ['plateau', 'poly'],
                  'Learning rate decay policy.')

flags.DEFINE_boolean('use_dpc', False, 'Use DPC architecture or not.')
flags.DEFINE_integer('decoder_stride', None, 'Decoder stride value.')

# Checkpoint related flags
flags.DEFINE_string('checkpoint_dir', None,
                    'Save directory path for checkpoint.')
flags.DEFINE_string('checkpoint_restore_path', None,
                    'Restore checkpoint path.')
flags.DEFINE_boolean('checkpoint_skip_mismatch', False,
                     'Skip mismatches in checkpoint restore or not.')
flags.DEFINE_string('checkpoint_note', '',
                    'A checkpoint note to add to the checkpoint name.')


def train(_):
    dataset_name = FLAGS.dataset
    dataset_train_split = FLAGS.dataset_train_split
    dataset_test_split = FLAGS.dataset_test_split
    dataset_dir = FLAGS.dataset_dir

    batch_size = FLAGS.batch_size
    input_size = FLAGS.crop_size
    num_epochs = FLAGS.num_epochs

    initial_lr = FLAGS.initial_lr

    use_dpc = FLAGS.use_dpc
    decoder_stride = FLAGS.decoder_stride

    checkpoint_base_path = FLAGS.checkpoint_dir
    checkpoint_restore_path = FLAGS.checkpoint_restore_path
    skip_mismatch = FLAGS.checkpoint_skip_mismatch
    checkpoint_note = FLAGS.checkpoint_note

    dataset, dataset_desc = get_dataset(dataset_name,
                                        dataset_train_split,
                                        dataset_dir,
                                        interleave=False)
    preprocessed_dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE).shuffle(1024).map(
            preprocess(input_size,
                       is_training=True,
                       ignore_label=dataset_desc.ignore_label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE).batch(
                    batch_size).repeat()

    TRAIN_LENGTH = dataset_desc.splits_to_sizes[dataset_train_split]
    STEPS_PER_EPOCH = TRAIN_LENGTH // batch_size

    VALIDATION_STEPS = dataset_desc.splits_to_sizes[
        dataset_test_split] // batch_size

    val_dataset, _ = get_dataset(dataset_name, dataset_test_split, dataset_dir)
    preprocessed_val_dataset = val_dataset.map(
        preprocess(input_size,
                   is_training=False,
                   ignore_label=dataset_desc.ignore_label)).batch(batch_size)

    # for im in preprocessed_val_dataset.take(1):
    #     for i in range(0, batch_size):
    #         plt.subplot(121).imshow(im[0][i, :])
    #         plt.subplot(122).imshow(tf.squeeze(im[1][i, :]),
    #                                 cmap='gray',
    #                                 vmin=0,
    #                                 vmax=92)
    #         plt.show()

    inputs = keras.Input(shape=(input_size[0], input_size[1], 3))
    outputs = shufflenet_v2_segmentation(
        inputs,
        dataset_desc.num_classes,
        filter_per_encoder_branch=256,
        use_dpc=use_dpc)
    model = keras.Model(inputs=inputs, outputs=outputs)

    train_loss = SoftmaxCrossEntropy(dataset_desc.num_classes,
                                     dataset_desc.ignore_label)
    train_metrics = [
        WeightedSparseMeanIoU(dataset_desc.num_classes,
                              dataset_desc.ignore_label),
        WeightedAccuracy(dataset_desc.num_classes, dataset_desc.ignore_label)
    ]

    learning_rate_fn = initial_lr
    callbacks = []

    if FLAGS.lr_decay_policy == 'plateau':
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.1,
                                                      patience=5,
                                                      min_lr=1e-8)
        callbacks.append(reduce_lr)
    elif FLAGS.lr_decay_policy == 'poly':
        learning_rate_fn = keras.optimizers.schedules.PolynomialDecay(
            initial_lr,
            round(dataset_desc.splits_to_sizes[dataset_train_split] /
                  batch_size) * num_epochs,
            0.0,
            power=0.9)

    if FLAGS.optimizer == 'adam':
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate_fn,
            # epsilon=0.001,
            amsgrad=True)
    elif FLAGS.optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate_fn,
                                         momentum=0.9)

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        f'{checkpoint_base_path}/' + f'{dataset_name}_{dataset_train_split}' +
        f'_ds{decoder_stride}_b{batch_size}' +
        f'_epoch{num_epochs}_{checkpoint_note}.h5',
        save_best_only=False,
        verbose=True)
    callbacks.append(checkpoint_callback)

    if checkpoint_restore_path is not None:
        model.load_weights(checkpoint_restore_path,
                           by_name=True,
                           skip_mismatch=skip_mismatch)

    model.compile(loss=train_loss, optimizer=optimizer, metrics=train_metrics)

    model.fit(preprocessed_dataset,
              epochs=num_epochs,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_data=preprocessed_val_dataset,
              validation_steps=VALIDATION_STEPS,
              callbacks=callbacks)


if __name__ == '__main__':
    flags.mark_flag_as_required('checkpoint_dir')
    flags.mark_flag_as_required('dataset_dir')
    app.run(train)
