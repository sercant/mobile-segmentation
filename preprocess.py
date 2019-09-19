import tensorflow as tf


@tf.function
def flip_dim(tensor_list: list, prob: float = 0.5, dim: int = 1):
    random_value = tf.random.uniform([])

    @tf.function
    def flip():
        flipped = []
        for tensor in tensor_list:
            flipped.append(tf.reverse(tensor, [dim]))
        return flipped

    outputs = tf.cond(pred=tf.less_equal(random_value, prob),
                      true_fn=flip,
                      false_fn=lambda: tensor_list)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    return outputs


@tf.function
def resize_with_pad(image: tf.Tensor,
                    target_height: int,
                    target_width: int,
                    pad_value: float,
                    method=tf.image.ResizeMethod.BILINEAR):
    _in_type = image.dtype
    assert _in_type in (tf.float32, tf.int32)

    image = image - pad_value
    image = tf.image.resize_with_pad(image,
                                     target_height,
                                     target_width,
                                     method=method)
    image = image + pad_value

    return tf.cast(image, _in_type)


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


def preprocess(input_size: list,
               image_pad_val: float = 127.5,
               is_training: bool = True,
               ignore_label: int = 255):
    @tf.function
    def _func(inputs: dict):
        image = tf.cast(inputs['image'], tf.float32)
        label = inputs['labels_class']
        if label is not None:
            label = tf.cast(label, tf.int32)

        if is_training:
            image, label = flip_dim([image, label])
        image, label = resize_to_range(image, label, input_size[0],
                                       input_size[1], image_pad_val,
                                       ignore_label)
        image = image / 255.0 - 1.0
        return image, label

    return _func
