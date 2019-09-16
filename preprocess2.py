import tensorflow as tf


@tf.function
def resize_with_pad(image: tf.Tensor,
                    target_height: int,
                    target_width: int,
                    pad_value: float,
                    method=tf.image.ResizeMethod.BILINEAR):
    image_type = image.dtype
    assert image_type == tf.float32 or image_type == tf.int32

    image = image - pad_value
    image = tf.image.resize_with_pad(image,
                                     target_height,
                                     target_width,
                                     method=method)
    image = image + pad_value

    return tf.cast(image, image_type)


@tf.function
def resize_to_range(image: tf.Tensor,
                    label: tf.Tensor = None,
                    target_height: int = 0,
                    target_width: int = 0,
                    image_pad_value: float = 127.5,
                    label_pad_value: int = 255):
    image = resize_with_pad(image, target_height, target_width,
                            image_pad_value)
    if label is not None:
        label = resize_with_pad(label,
                                target_height,
                                target_width,
                                label_pad_value,
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image, tf.cast(label, tf.int32)


def preprocess(dataset_desc: dict,
               input_size: list,
               image_pad_val: float = 127.5):
    @tf.function
    def _func(inputs: dict):
        image = tf.cast(inputs['image'], tf.float32)
        label = inputs['labels_class']
        if label is not None:
            label = tf.cast(label, tf.int32)

        return resize_to_range(image, label, input_size[0], input_size[1],
                               image_pad_val, dataset_desc.ignore_label)

    return _func
