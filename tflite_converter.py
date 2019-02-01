
import tensorflow as tf
import utils.load_env
import common
import model

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

if __name__ == '__main__':
    input_tensor_name = 'input_0'
    input_size = [1, 257, 257, 3]
    outputs_to_num_classes = {common.OUTPUT_TYPE: 19}

    model_options = common.ModelOptions(
        outputs_to_num_classes=outputs_to_num_classes,
        crop_size=None,
        atrous_rates=None,
        preprocess_images=True,
        output_stride=16)

    g = tf.Graph()
    with g.as_default():
        with tf.Session(graph=g) as sess:
            inputs = tf.placeholder(
                tf.float32, input_size, name=input_tensor_name)
            outputs_to_scales_to_logits = model.predict_labels_tflite(
                inputs,
                model_options=model_options)
            predictions = outputs_to_scales_to_logits[common.OUTPUT_TYPE]
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

            with tf.Graph().as_default() as graph:
                transforms = [
                    'strip_unused_nodes(type=float,shape=\"{}\")'.format(
                        ','.join(['{}'.format(s) for s in input_size])),
                    'remove_nodes(op=Identity, op=CheckNumerics)',
                    'flatten_atrous_conv',
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

                with tf.gfile.GFile('dist/tensorflow_graph.pb', "wb") as f:
                    f.write(constant_graph.SerializeToString())

                tf.import_graph_def(constant_graph, name='')

                x = graph.get_tensor_by_name('{}:0'.format(input_tensor_name))
                y = graph.get_tensor_by_name('{}:0'.format(output_tensor_name))
                # y.set_shape([1, 224, 224, 3])

            run_meta = tf.RunMetadata()
            with tf.Session(graph=graph) as sess:
                converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [
                    x], [y])
                tflite_model = converter.convert()
                open("dist/tflite_graph.tflite", "wb").write(tflite_model)

                opts = tf.profiler.ProfileOptionBuilder.float_operation()
                flops = tf.profiler.profile(
                    graph, run_meta=run_meta, cmd='op', options=opts)
                print('FLOPS: ', flops.total_float_ops)
