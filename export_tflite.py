
import tensorflow as tf
import common
import model

flags = tf.app.flags

FLAGS = flags.FLAGS

INPUT_TENSOR_NAME = 'input_0'
INPUT_SIZE = [1, 225, 225, 3]
NUMBER_OF_CLASSES = 4
OUTPUT_STRIDE = 16

MODEL_VARIANT = 'shufflenet_v2'
USE_DPC = False
CHECKPOINT_PATH = './logs'

OUT_PATH_TFLITE = 'dist/tflite_graph.tflite'
OUT_PATH_FROZEN_GRAPH = 'dist/tensorflow_graph.pb'

if __name__ == '__main__':
    input_tensor_name = INPUT_TENSOR_NAME
    input_size = INPUT_SIZE
    outputs_to_num_classes = {common.OUTPUT_TYPE: NUMBER_OF_CLASSES}

    FLAGS.model_variant = MODEL_VARIANT
    FLAGS.dense_prediction_cell_json = './core/dense_prediction_cell_branch5_top1_cityscapes.json' if USE_DPC else ''
    chkpt_path = CHECKPOINT_PATH

    model_options = common.ModelOptions(
        outputs_to_num_classes=outputs_to_num_classes,
        crop_size=input_size[1:3],
        atrous_rates=None,
        output_stride=OUTPUT_STRIDE)

    g = tf.Graph()
    with g.as_default():
        with tf.Session(graph=g) as sess:
            inputs = tf.placeholder(
                tf.float32, input_size, name=input_tensor_name)
            outputs_to_scales_to_logits = model.predict_labels(
                inputs,
                model_options=model_options)
            predictions = tf.cast(
                outputs_to_scales_to_logits[common.OUTPUT_TYPE], tf.int32)
            output_tensor_name = predictions.name.split(':')[0]

            sess.run(tf.global_variables_initializer())
            if chkpt_path:
                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(chkpt_path))

            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                # The graph_def is used to retrieve the nodes
                tf.get_default_graph().as_graph_def(),
                # The output node names are used to select the usefull nodes
                [output_tensor_name]
            )
            with tf.gfile.GFile(OUT_PATH_FROZEN_GRAPH, "wb") as f:
                f.write(constant_graph.SerializeToString())

            with tf.Graph().as_default() as graph:
                transforms = [
                    'strip_unused_nodes(type=float,shape=\"{}\")'.format(
                        ','.join(['{}'.format(s) for s in input_size])),
                    'remove_nodes(op=Identity, op=CheckNumerics)',
                    # 'flatten_atrous_conv',
                    'fold_constants(ignore_errors=true)',
                    'fold_batch_norms',
                    'fold_old_batch_norms'
                ]
                constant_graph = tf.tools.graph_transforms.TransformGraph(
                    constant_graph,
                    ['{}:0'.format(input_tensor_name)],
                    ['{}:0'.format(output_tensor_name)],
                    transforms
                )
                tf.import_graph_def(constant_graph, name='')

                x = graph.get_tensor_by_name('{}:0'.format(input_tensor_name))
                y = graph.get_tensor_by_name('{}:0'.format(output_tensor_name))

            with tf.Session(graph=graph) as sess:
                converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [
                    x], [y])
                tflite_model = converter.convert()
                open(OUT_PATH_TFLITE, "wb").write(tflite_model)
