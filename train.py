import tensorflow as tf
from tensorflow import keras

from dataset import get_dataset
from preprocess import preprocess
from model import shufflenet_v2_segmentation
from utils.losses import SoftmaxCrossEntropy
from utils.metrics import WeightedSparseMeanIoU, WeightedAccuracy


def train():
    dataset_name = 'cityscapes'
    dataset_dir = './data/research/cityscapes/tfrecord'
    dataset_split = 'train_extra'
    batch_size = 16
    input_size = [769, 769]
    use_dpc = False
    decoder_stride = None
    num_epochs = 60
    initial_lr = 0.001
    small_backend = False
    checkpoint_base_path = './data/research/cityscapes/checkpoints'
    restore_weights_from = f'{checkpoint_base_path}/cityscapes_train_extra_full_dsNone_b16_epoch60_adam_se_v2resize.h5'
    note = f'{"small" if small_backend else "full"}_adam_se'

    dataset, dataset_desc = get_dataset(dataset_name,
                                        dataset_split,
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

    TRAIN_LENGTH = dataset_desc.splits_to_sizes[dataset_split]
    STEPS_PER_EPOCH = TRAIN_LENGTH // batch_size

    VALIDATION_STEPS = dataset_desc.splits_to_sizes['val'] // batch_size

    val_dataset, _ = get_dataset(dataset_name, 'val', dataset_dir)
    preprocessed_val_dataset = val_dataset.map(
        preprocess(input_size,
                   is_training=False,
                   ignore_label=dataset_desc.ignore_label)).batch(batch_size)

    inputs = keras.Input(shape=(input_size[0], input_size[1], 3))
    outputs = shufflenet_v2_segmentation(
        inputs,
        dataset_desc.num_classes,
        filter_per_encoder_branch=128 if small_backend else 256,
        use_dpc=use_dpc,
        small_backend=small_backend)
    model = keras.Model(inputs=inputs, outputs=outputs)

    train_loss = SoftmaxCrossEntropy(dataset_desc.num_classes,
                                     dataset_desc.ignore_label)
    train_metrics = [
        WeightedSparseMeanIoU(dataset_desc.num_classes,
                              dataset_desc.ignore_label),
        WeightedAccuracy(dataset_desc.num_classes, dataset_desc.ignore_label)
    ]

    learning_rate_fn = initial_lr
    # learning_rate_fn = keras.optimizers.schedules.PolynomialDecay(
    #    initial_lr,
    #    round(dataset_desc.splits_to_sizes[dataset_split] / batch_size) *
    #    num_epochs,
    #    0.0,
    #    power=0.9)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.1,
                                                  patience=5,
                                                  min_lr=1e-8)

    # optimizer = keras.optimizers.SGD(learning_rate=learning_rate_fn,
    #                                    momentum=0.9)
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate_fn,
        # epsilon=0.001,
        amsgrad=True)

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        f'{checkpoint_base_path}/' + f'{dataset_name}_{dataset_split}' +
        f'_ds{decoder_stride}_b{batch_size}' + f'_epoch{num_epochs}_{note}.h5',
        save_best_only=False,
        verbose=True)

    callbacks = [model_checkpoint, reduce_lr]

    if restore_weights_from is not None:
        model.load_weights(restore_weights_from, by_name=True)

    model.compile(loss=train_loss, optimizer=optimizer, metrics=train_metrics)

    model.fit(preprocessed_dataset,
              epochs=num_epochs,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_data=preprocessed_val_dataset,
              validation_steps=VALIDATION_STEPS,
              callbacks=callbacks)


if __name__ == '__main__':
    train()
