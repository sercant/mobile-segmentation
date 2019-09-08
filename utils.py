import tensorflow as tf
from tensorflow.python import keras as K
from tensorflow.python.keras.saving.hdf5_format import *


def _legacy_weights(model):
    """DO NOT USE.
  For legacy reason, the model.weights was in the order of
  [self.trainable_weights + self.non_trainable_weights], and this order was
  used for preserving the weights in h5 format. The new order of model.weights
  are the same as model.get_weights() which is more intuitive for user. To
  keep supporting the existing saved h5 file, this method should be used to
  save/load weights. In future version, we will delete this method and
  introduce a breaking change for h5 and stay with the new order for weights.
  Args:
    model: a model or layer instance.
  Returns:
    A list of variables with the order of trainable_weights, followed by
      non_trainable_weights.
  """
    return model.trainable_weights + model.non_trainable_weights


def load_weights_from_hdf5_group_by_name(f, layers, skip_mismatch=False):
    """Implements name-based weight loading.
  (instead of topological weight loading).
  Layers that have no matching name are skipped.
  Arguments:
      f: A pointer to a HDF5 group.
      layers: a list of target layers.
  Raises:
      ValueError: in case of mismatch between provided layers
          and weights file.
  """
    if 'keras_version' in f.attrs:
        original_keras_version = f.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f.attrs:
        original_backend = f.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    # New file format.
    layer_names = load_attributes_from_hdf5_group(f, 'layer_names')

    # Reverse index of layer name to list of layers with name.
    index = {}
    for layer in layers:
        if layer.name:
            index.setdefault(layer.name, []).append(layer)

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, name in enumerate(layer_names):
        g = f[name]
        weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
        weight_values = [
            np.asarray(g[weight_name]) for weight_name in weight_names
        ]

        for layer in index.get(name, []):
            symbolic_weights = _legacy_weights(layer)
            weight_values = preprocess_weights_for_loading(
                layer, weight_values, original_keras_version, original_backend)
            if len(weight_values) != len(symbolic_weights):
                if skip_mismatch:
                    tf.get_logger().warn('Layer #' + str(k) + ' (named "' +
                                         layer.name + '") expects ' +
                                         str(len(symbolic_weights)) +
                                         ' weight(s), but the saved weights' +
                                         ' have ' + str(len(weight_values)) +
                                         ' element(s).')
                    continue
                else:
                    raise ValueError('Layer #' + str(k) + ' (named "' +
                                     layer.name + '") expects ' +
                                     str(len(symbolic_weights)) +
                                     ' weight(s), but the saved weights' +
                                     ' have ' + str(len(weight_values)) +
                                     ' element(s).')
            # Set values.
            for i in range(len(weight_values)):
                if K.int_shape(symbolic_weights[i]) != weight_values[i].shape:
                    if skip_mismatch:
                        tf.get_logger().warn(
                            'Layer #' + str(k) + ' (named "' + layer.name +
                            '") has shape {}'.format(
                                K.int_shape(symbolic_weights[i])) +
                            ', but the saved weight has shape ' +
                            str(weight_values[i].shape) + '.')
                        continue
                    else:
                        raise ValueError(
                            'Layer #' + str(k) + ' (named "' + layer.name +
                            '"), weight ' + str(symbolic_weights[i]) +
                            ' has shape {}'.format(
                                K.int_shape(symbolic_weights[i])) +
                            ', but the saved weight has shape ' +
                            str(weight_values[i].shape) + '.')

                else:
                    weight_value_tuples.append(
                        (symbolic_weights[i], weight_values[i]))
    K.batch_set_value(weight_value_tuples)
