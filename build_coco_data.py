import io
import json
import os
import sys
import math
import numpy as np
import PIL.Image
from pycocotools import mask
import tensorflow as tf

sys.path.append(os.getcwd() + '/tf_models/research')
sys.path.append(os.getcwd() + '/tf_models/research/deeplab/')
sys.path.append(os.getcwd() + '/tf_models/research/deeplab/datasets')
sys.path.append(os.getcwd() + '/tf_models/research/slim')
try:
    import build_data
except:
    print('Can\'t import deeplab libraries!')
    raise

flags = tf.app.flags
tf.flags.DEFINE_string('dataset_dir', './data/coco',
                       'Directory of the coco dataset.')
tf.flags.DEFINE_string('output_dir', './data/records',
                       'Output data directory.')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

_NUM_SHARDS = 4


def getCatIds(annotations, catNms=[], supNms=[], catIds=[]):
    """
    filtering parameters. default skips that filter.
    :param catNms (str array)  : get cats for given cat names
    :param supNms (str array)  : get cats for given supercategory names
    :param catIds (int array)  : get cats for given cat ids
    :return: ids (int array)   : integer array of cat ids
    """
    # catNms = catNms if _isArrayLike(catNms) else [catNms]
    # supNms = supNms if _isArrayLike(supNms) else [supNms]
    # catIds = catIds if _isArrayLike(catIds) else [catIds]

    if not catNms and not supNms and not catIds:
        cats = annotations['categories']
    else:
        cats = annotations['categories']
        cats = cats if not catNms else [
            cat for cat in cats if cat['name'] in catNms]
        cats = cats if not supNms else [
            cat for cat in cats if cat['supercategory'] in supNms]
        cats = cats if not catIds else [
            cat for cat in cats if cat['id'] in catIds]
    ids = [cat['id'] for cat in cats]
    return ids


def _convert_dataset(dataset_split, dataset_dir, cat_nms=['apple', 'book', 'keyboard']):
    """Converts the ADE20k dataset into into tfrecord format.

    Args:
      dataset_split: Dataset split (e.g., train, val).
      dataset_dir: Dir in which the dataset locates.
      dataset_label_dir: Dir in which the annotations locates.

    Raises:
      RuntimeError: If loaded image and label have different shape.
    """

    with tf.gfile.GFile(dataset_dir + '/annotations/instances_{}2017.json'.format(dataset_split), 'r') as fid:
        groundtruth_data = json.load(fid)
        images = groundtruth_data['images']

        cat_ids = getCatIds(groundtruth_data, catNms=cat_nms)
        class_ids = {}
        for i, cat_id in enumerate(cat_ids):
            class_ids[cat_id] = i + 1

        annotations_index = {}
        if 'annotations' in groundtruth_data:
            tf.logging.info(
                'Found groundtruth annotations. Building annotations index.')
            for annotation in groundtruth_data['annotations']:
                if annotation['category_id'] not in cat_ids:
                    continue
                image_id = annotation['image_id']
                if image_id not in annotations_index:
                    annotations_index[image_id] = []
                annotations_index[image_id].append(annotation)

        filtered_images = []
        for idx, image in enumerate(images):
            if image['id'] not in annotations_index.keys():
                continue
            annotations_list = annotations_index[image['id']]
            if annotations_list:
                filtered_images.append(image)
        images = filtered_images

        missing_annotation_count = 0
        for image in images:
            image_id = image['id']
            if image_id not in annotations_index:
                missing_annotation_count += 1
                annotations_index[image_id] = []
        tf.logging.info('%d images are missing annotations.',
                        missing_annotation_count)

    num_images = len(images)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = build_data.ImageReader('jpeg', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)
    # os.path.join(dataset_dir + '/{}2017'.format(dataset_split), img['file_name'])
    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            FLAGS.output_dir,
            '%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, _NUM_SHARDS))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_images, shard_id))
                sys.stdout.flush()
                # Read the image.
                img = images[i]
                image_filename = os.path.join(
                    dataset_dir + '/{}2017'.format(dataset_split), img['file_name'])

                # image_data = tf.gfile.GFile(image_filename, 'rb').read()
                image_p = PIL.Image.open(image_filename)
                output_io_img = io.BytesIO()
                image_p.save(output_io_img, format='JPEG')
                image_data = output_io_img.getvalue()
                height, width = image_reader.read_image_dims(image_data)

                # Read the semantic segmentation annotation.
                # prep seg data
                annotations = annotations_index[img['id']]
                segmented = np.zeros((height, width), np.uint8)
                for an in annotations:
                    run_len_encoding = mask.frPyObjects(an['segmentation'],
                                                        height, width)
                    binary_mask = mask.decode(run_len_encoding)
                    if not an['iscrowd']:
                        binary_mask = np.amax(binary_mask, axis=2)
                    segmented[binary_mask == 1] = class_ids[an['category_id']]
                seg_img = PIL.Image.fromarray(segmented, mode='L')
                output_io = io.BytesIO()
                seg_img.save(output_io, format='PNG')
                seg_data = output_io.getvalue()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)

                assert height == seg_height and width == seg_width

                example = build_data.image_seg_to_tfexample(
                    image_data, image_filename, height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()


def main(_):
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    _convert_dataset('val', FLAGS.dataset_dir)
    _convert_dataset('train', FLAGS.dataset_dir)


if __name__ == '__main__':
    tf.app.run()
