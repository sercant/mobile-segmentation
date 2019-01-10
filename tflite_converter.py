
import tensorflow as tf
import common
import model

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

if __name__ == '__main__':
    input_tensor_name = 'input_0'
    input_size = [1, 224, 224, 3]
    outputs_to_num_classes = {common.OUTPUT_TYPE: 4}

    model_options = common.ModelOptions(
        outputs_to_num_classes=outputs_to_num_classes,
        crop_size=input_size[1:3],
        atrous_rates=None,
        output_stride=16)

    g = tf.Graph()
    with g.as_default():
        with tf.Session(graph=g) as sess:
            inputs = tf.placeholder(
                tf.float32, input_size, name=input_tensor_name)
            outputs_to_scales_to_logits = model.predict_labels(
                inputs,
                model_options=model_options)
            predictions = tf.to_int32(
                outputs_to_scales_to_logits[common.OUTPUT_TYPE])
            output_tensor_name = predictions.name.split(':')[0]

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('./logs'))

            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                # The graph_def is used to retrieve the nodes
                tf.get_default_graph().as_graph_def(),
                # The output node names are used to select the usefull nodes
                [output_tensor_name]
            )
            with tf.gfile.GFile('dist/tensorflow_graph.pb', "wb") as f:
                f.write(constant_graph.SerializeToString())

            with tf.Graph().as_default() as graph:
                transforms = [
                    'strip_unused_nodes(type=int,shape=\"1,224,224,3\")',
                    'remove_nodes(op=Identity, op=CheckNumerics)',
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
                # y.set_shape([1, 224, 224, 3])

            with tf.Session(graph=graph) as sess:
                converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [
                    x], [y])
                tflite_model = converter.convert()
                open("dist/tflite_graph.tflite", "wb").write(tflite_model)
