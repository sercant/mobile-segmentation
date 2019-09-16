import os
import collections
import tensorflow as tf


@tf.function
def decode_image(x: tf.Tensor):
    return tf.io.decode_image(x, channels=3)


@tf.function
def decode_label(x: tf.Tensor):
    return tf.io.decode_image(x, channels=1)


@tf.function
def decode_tensor(x: tf.Tensor):
    return x


# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        'splits_to_sizes',  # Splits of the dataset into training, val, and test.
        'num_classes',  # Number of semantic classes, including the background
        # class (if exists). For example, there are 20
        # foreground classes + 1 background class in the PASCAL
        # VOC 2012 dataset. Thus, we set num_classes=21.
        'ignore_label',  # Ignore label value.
    ])

_CITYSCAPES_INFORMATION = DatasetDescriptor(splits_to_sizes={
    'train_extra': 19998,
    'train': 2975,
    'val': 500,
    'test': 1525
},
                                            num_classes=19,
                                            ignore_label=255)

_PASCAL_VOC_SEG_INFORMATION = DatasetDescriptor(splits_to_sizes={
    'train': 1464,
    'train_aug': 10582,
    'trainval': 2913,
    'val': 1449
},
                                                num_classes=21,
                                                ignore_label=255)

# These number (i.e., 'train'/'test') seems to have to be hard coded
# You are required to figure it out for your training/testing example.
_ADE20K_INFORMATION = DatasetDescriptor(splits_to_sizes={
    'train': 20210,
    'val': 2000
},
                                        num_classes=151,
                                        ignore_label=0)

_COCO_INFORMATION = DatasetDescriptor(splits_to_sizes={
    'train': 117266,
    'val': 4952
},
                                      num_classes=92,
                                      ignore_label=255)

_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'coco': _COCO_INFORMATION,
}

# Default file pattern of TFRecord of TensorFlow Example.
FILE_PATTERN = '%s-*'

# Specify how the TF-Examples are decoded.
FEATURE_DESCRIPTION = {
    'image/encoded':
    tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/filename':
    tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/format':
    tf.io.FixedLenFeature([], tf.string, default_value='jpeg'),
    'image/height':
    tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/width':
    tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/segmentation/class/encoded':
    tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/segmentation/class/format':
    tf.io.FixedLenFeature([], tf.string, default_value='png')
}

ITEMS_TO_HANDLERS = {
    'image': (decode_image, 'image/encoded'),
    'image_name': (decode_tensor, 'image/filename'),
    'height': (decode_tensor, 'image/height'),
    'width': (decode_tensor, 'image/width'),
    'labels_class': (decode_label, 'image/segmentation/class/encoded')
}


def get_dataset(dataset_name: str, split_name: str, dataset_dir: str):
    if dataset_name not in _DATASETS_INFORMATION:
        raise ValueError('The specified dataset is not supported yet.')

    @tf.function
    def _parse_to_items(example_proto: tf.Tensor):
        example: dict = tf.io.parse_single_example(example_proto,
                                                   FEATURE_DESCRIPTION)
        inputs = {}
        for key, tup in ITEMS_TO_HANDLERS.items():
            decoder_func, internal_key = tup
            inputs[key] = decoder_func(example[internal_key])
        return inputs

    files = tf.io.matching_files(
        os.path.join(dataset_dir, FILE_PATTERN % split_name))

    _dataset = tf.data.TFRecordDataset(files)
    _dataset = _dataset.map(_parse_to_items)

    return _dataset, _DATASETS_INFORMATION[dataset_name]


if __name__ == "__main__":
    dataset, dataset_description = get_dataset("coco", "val", "./data/coco/tfrecord")

    for item in dataset.take(10):
        print(item)
