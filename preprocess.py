import tensorflow as tf


@tf.function
def get_random_scale(min_val: float, max_val: float, step: float):
    if min_val < 0 or min_val > max_val:
        raise ValueError('Unexpected value of min_val.')

    if min_val == max_val:
        return tf.cast(min_val, tf.float32)

    # When step = 0, we sample the value uniformly from [min, max).
    if step == 0:
        return tf.random.uniform([], minval=min_val, maxval=max_val)

    # When step != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_val - min_val) / step + 1)
    scale_factors = tf.linspace(min_val, max_val, num_steps)
    shuffled_scale_factors = tf.random.shuffle(scale_factors)
    return shuffled_scale_factors[0]


@tf.function
def scale(image: tf.Tensor, label: tf.Tensor, scale: float):
    if scale == 1.0:
        return image, label

    im_shape = tf.cast(tf.shape(image), tf.float32)
    _target_height, _target_width = tf.cast(im_shape[0] * scale,
                                            tf.int32), tf.cast(
                                                im_shape[1] * scale, tf.int32)
    image = tf.image.resize_with_pad(image,
                                     _target_height,
                                     _target_width,
                                     method="bilinear")
    if label is not None:
        label = tf.image.resize_with_pad(label,
                                         _target_height,
                                         _target_width,
                                         method="nearest")

    return image, label


@tf.function
def pad_to_bounding_box(image: tf.Tensor, target_height: int,
                        target_width: int, value: float):
    image = image - value
    image = tf.image.pad_to_bounding_box(image, 0, 0, target_height,
                                         target_width)
    image = image + value

    return image


@tf.function
def random_crop(image: tf.Tensor, label: tf.Tensor, crop_height: int,
                crop_width: int):
    im_shape = tf.shape(image)

    max_offset_height = tf.cast(im_shape[0] - crop_height + 1, tf.int32)
    max_offset_width = tf.cast(im_shape[1] - crop_width + 1, tf.int32)

    offset_height = tf.random.uniform([],
                                      maxval=max_offset_height,
                                      dtype=tf.int32)
    offset_width = tf.random.uniform([],
                                     maxval=max_offset_width,
                                     dtype=tf.int32)

    @tf.function
    def _crop(image: tf.Tensor, offset_height: int, offset_width: int,
              target_height: int, target_width: int):
        if image is None:
            return None
        return tf.image.crop_to_bounding_box(image, offset_height,
                                             offset_width, target_height,
                                             target_width)

    return _crop(image, offset_height, offset_width, crop_height,
                 crop_width), _crop(label, offset_height, offset_width,
                                    crop_height, crop_width)


@tf.function
def _preprocess(image: tf.Tensor, label: tf.Tensor, crop_height: int,
                crop_width: int, min_scale: float, max_scale: float,
                scale_step: float, mean_pixel_val: (float, list),
                ignore_label: int, is_training: bool):
    if is_training:
        rand_scale = get_random_scale(min_scale, max_scale, scale_step)
        image, label = scale(image, label, rand_scale)

        im_shape = tf.shape(image)
        target_height = im_shape[0] + tf.maximum(crop_height - im_shape[0], 0)
        target_width = im_shape[1] + tf.maximum(crop_width - im_shape[1], 0)

        image = pad_to_bounding_box(image, target_height, target_width,
                                    mean_pixel_val)
        if label is not None:
            label = pad_to_bounding_box(label, target_height, target_width,
                                        ignore_label)

        image, label = random_crop(image, label, crop_height, crop_width)

        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            if label is not None:
                label = tf.image.flip_left_right(label)
    else:
        image_shape = tf.shape(image)
        if image_shape[1] > crop_height or image_shape[2] > crop_width:
            image = image - mean_pixel_val
            image = tf.image.resize_with_pad(image,
                                             crop_height,
                                             crop_width,
                                             method="bilinear")
            image = image + mean_pixel_val
            if label is not None:
                label = label - ignore_label
                label = tf.image.resize_with_pad(label,
                                                 crop_height,
                                                 crop_width,
                                                 method="nearest")
                label = label + ignore_label
        else:
            image = pad_to_bounding_box(image, crop_height, crop_width,
                                        mean_pixel_val)
            if label is not None:
                label = pad_to_bounding_box(label, crop_height, crop_width,
                                            ignore_label)

    return image, label


def preprocess(input_size: list,
               image_pad_val: float = 127.5,
               min_scale: float = 0.5,
               max_scale: float = 2.0,
               scale_step: float = 0.25,
               is_training: bool = True,
               ignore_label: int = 255):
    @tf.function
    def _func(inputs: dict):
        image = tf.cast(inputs['image'], tf.float32)
        label = inputs['labels_class']

        if label is not None:
            label = tf.cast(label, tf.int32)

        image, label = _preprocess(image, label, input_size[0], input_size[1],
                                   min_scale, max_scale, scale_step,
                                   image_pad_val, ignore_label, is_training)
        image = (image / 127.5) - 1.0

        return image, label

    return _func
