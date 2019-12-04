import tensorflow as tf
from tensorflow import keras

from dataset import get_dataset
from preprocess import preprocess
from model import shufflenet_v2_segmentation
from utils.losses import SoftmaxCrossEntropy
from utils.metrics import WeightedSparseMeanIoU, WeightedAccuracy


def train():
    dataset_name = "cityscapes"
    dataset_dir = "./data/research/cityscapes/tfrecord"
    dataset_split = "train_extra"
    batch_size = 16
    input_size = [769, 769]
    use_dpc = False
    decoder_stride = None
    num_epochs = 40
    initial_lr = 0.01
    small_backend = False
    restore_weights_from = './checkpoints/cifar100_shufflenet_v2_51_80.h5'

    dataset, dataset_desc = get_dataset(dataset_name, dataset_split,
                                        dataset_dir)
    preprocessed_dataset = dataset.map(
        preprocess(input_size,
                   is_training=True,
                   ignore_label=dataset_desc.ignore_label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    preprocessed_dataset = preprocessed_dataset.shuffle(512).batch(
        batch_size).repeat()
    preprocessed_dataset = preprocessed_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

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

    train_loss = SoftmaxCrossEntropy(dataset_desc.num_classes, dataset_desc.ignore_label)
    train_metrics = [
        WeightedSparseMeanIoU(dataset_desc.num_classes,
                              dataset_desc.ignore_label),
        WeightedAccuracy(dataset_desc.ignore_label)
    ]

    learning_rate_fn = keras.optimizers.schedules.PolynomialDecay(
        initial_lr,
        round(dataset_desc.splits_to_sizes[dataset_split] / batch_size) *
        num_epochs,
        0.0,
        power=0.9)
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate_fn,
                                     momentum=0.9)

    model_checkpoint = keras.callbacks.ModelCheckpoint(
        './data/research/cityscapes/checkpoints/{}_{}_{}_ds{}_b{}_epoch{}_new_loss.h5'
        .format(dataset_name, dataset_split,
                'small' if small_backend else 'full', decoder_stride,
                batch_size, num_epochs),
        save_best_only=True,
        verbose=True)

    if restore_weights_from is not None:
        model.load_weights(restore_weights_from, by_name=True)

    model.compile(loss=train_loss, optimizer=optimizer, metrics=train_metrics)

    model.fit(preprocessed_dataset,
              epochs=num_epochs,
              steps_per_epoch=STEPS_PER_EPOCH,
              validation_data=preprocessed_val_dataset,
              validation_steps=VALIDATION_STEPS,
              callbacks=[model_checkpoint])


if __name__ == "__main__":
    train()
