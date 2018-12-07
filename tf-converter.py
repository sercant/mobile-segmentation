import os

import keras
import tensorflow as tf
from tensorflow.python.framework.graph_util_impl import convert_variables_to_constants
from keras import backend as K
from keras import layers
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.tools.graph_transforms import TransformGraph

from nets.deeplabv3p import network

transforms = [
    'strip_unused_nodes(type=float,shape=\"1,224,224,3\")',
    'remove_nodes(op=Identity, op=CheckNumerics)',
    'fold_constants(ignore_errors=true)',
    'fold_batch_norms',
    'fold_old_batch_norms'
]

if __name__ == "__main__":
    K.set_learning_phase(0)

    input_tensor = layers.Input(
        shape=(224, 224, 3), name='input_1')

    model = network(
        input_tensor=input_tensor,
        num_classes=3
    )

    # model = keras.models.Model(input_tensor, net)

    # Load weights.
    model.load_weights('./dist/weights_deep.h5')

    # import skimage
    # input_data = skimage.io.imread('data/keyboard.jpg')
    # input_data = skimage.transform.resize(input_data, (224, 224))
    # input_data = np.expand_dims(input_data, axis=0)
    # # model.summary()
    # y = model.predict(input_data)
    # for i in range(3):
    #     plt.imshow(y[0,:,:,i])
    #     plt.show()

    pred_node_names = ['output_%s' % n for n in range(1)]
    print('output nodes names are: ', pred_node_names)

    for idx, name in enumerate(pred_node_names):
        tf.identity(model.output[idx], name=name)

    sess = K.get_session()
    constant_graph = convert_variables_to_constants(sess,
                                                    sess.graph.as_graph_def(),
                                                    pred_node_names)

    prefix = 'coco'

    with tf.Graph().as_default() as graph:
        constant_graph = TransformGraph(
            constant_graph,
            ['input_1:0'],
            ['output_1/ResizeBilinear:0'],
            transforms
        )
        tf.import_graph_def(constant_graph, name=prefix)

    for op in graph.get_operations():
        print(op.name)

    x = graph.get_tensor_by_name('%s/input_1:0' % prefix)
    y = graph.get_tensor_by_name('%s/output_1/ResizeBilinear:0' % prefix)

    with tf.Session(graph=graph) as sess:
        converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [x], [y])
        tflite_model = converter.convert()
        open("dist/converted_model.tflite", "wb").write(tflite_model)

        # Load TFLite model and allocate tensors.
        interpreter = tf.contrib.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # input_data = np.random.random([1, 224, 224, 3]).astype(np.float32)
        import skimage
        input_data = skimage.io.imread('data/keyboard.jpg')
        input_data = skimage.transform.resize(input_data, (224, 224))
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        plt.imshow(output_data)
        plt.show()

