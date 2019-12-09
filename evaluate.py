import tensorflow as tf
from tensorflow import keras

from dataset import get_dataset
from preprocess import preprocess
from model import shufflenet_v2_segmentation
from utils.losses import SoftmaxCrossEntropy, LovaszSoftmax
from utils.metrics import WeightedSparseMeanIoU


def evaluate():
    dataset_name = "cityscapes"
    dataset_dir = "./data/research/cityscapes/tfrecord"
    dataset_split = "val"
    batch_size = 10
    input_size = [1025, 2049]
    use_dpc = False
    decoder_stride = None
    small_backend = False
    restore_weights_from = './data/research/cityscapes/checkpoints/cityscapes_train_extra_full_dsNone_b16_epoch60_adam_se.h5'

    val_dataset, dataset_desc = get_dataset(dataset_name, dataset_split,
                                            dataset_dir)
    preprocessed_val_dataset = val_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE).map(
            preprocess(input_size,
                       is_training=False,
                       ignore_label=dataset_desc.ignore_label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
                    batch_size)

    inputs = keras.Input(shape=(input_size[0], input_size[1], 3))
    outputs = shufflenet_v2_segmentation(
        inputs,
        dataset_desc.num_classes,
        filter_per_encoder_branch=128 if small_backend else 256,
        use_dpc=use_dpc,
        output_stride=16,
        small_backend=small_backend)
    model = keras.Model(inputs=inputs, outputs=outputs)
    if restore_weights_from is not None:
        model.load_weights(restore_weights_from)

    train_loss = SoftmaxCrossEntropy(dataset_desc.num_classes,
                                     dataset_desc.ignore_label)
    train_accuracy = WeightedSparseMeanIoU(dataset_desc.num_classes,
                                           dataset_desc.ignore_label)

    model.compile(loss=train_loss, metrics=[train_accuracy])
    model.evaluate(preprocessed_val_dataset,
                   steps=dataset_desc.splits_to_sizes[dataset_split] //
                   batch_size)


if __name__ == "__main__":
    evaluate()
