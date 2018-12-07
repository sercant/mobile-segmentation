# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for DeepLab model and some helper functions."""

import tensorflow as tf

from deeplab import common
from deeplab import model


class DeeplabModelTest(tf.test.TestCase):

#   def testWrongDeepLabVariant(self):
#     model_options = common.ModelOptions([])._replace(
#         model_variant='no_such_variant')
#     with self.assertRaises(ValueError):
#       model._get_logits(images=[], model_options=model_options)

#   def testBuildDeepLabv2(self):
#     batch_size = 2
#     crop_size = [41, 41]

#     # Test with two image_pyramids.
#     image_pyramids = [[1], [0.5, 1]]

#     # Test two model variants.
#     model_variants = ['xception_65', 'mobilenet_v2']

#     # Test with two output_types.
#     outputs_to_num_classes = {'semantic': 3,
#                               'direction': 2}

#     expected_endpoints = [['merged_logits'],
#                           ['merged_logits',
#                            'logits_0.50',
#                            'logits_1.00']]
#     expected_num_logits = [1, 3]

#     for model_variant in model_variants:
#       model_options = common.ModelOptions(outputs_to_num_classes)._replace(
#           add_image_level_feature=False,
#           aspp_with_batch_norm=False,
#           aspp_with_separable_conv=False,
#           model_variant=model_variant)

#       for i, image_pyramid in enumerate(image_pyramids):
#         g = tf.Graph()
#         with g.as_default():
#           with self.test_session(graph=g):
#             inputs = tf.random_uniform(
#                 (batch_size, crop_size[0], crop_size[1], 3))
#             outputs_to_scales_to_logits = model.multi_scale_logits(
#                 inputs, model_options, image_pyramid=image_pyramid)

#             # Check computed results for each output type.
#             for output in outputs_to_num_classes:
#               scales_to_logits = outputs_to_scales_to_logits[output]
#               self.assertListEqual(sorted(scales_to_logits.keys()),
#                                    sorted(expected_endpoints[i]))

#               # Expected number of logits = len(image_pyramid) + 1, since the
#               # last logits is merged from all the scales.
#               self.assertEqual(len(scales_to_logits), expected_num_logits[i])

  def testForwardpassDeepLabv3plus(self):
    crop_size = [33, 33]
    outputs_to_num_classes = {'semantic': 3}

    model_options = common.ModelOptions(
        outputs_to_num_classes,
        crop_size,
        output_stride=16
    )._replace(
        add_image_level_feature=True,
        aspp_with_batch_norm=True,
        logits_kernel_size=1,
        model_variant='mobilenet_v2')  # Employ MobileNetv2 for fast test.

    g = tf.Graph()
    with g.as_default():
      with self.test_session(graph=g) as sess:
        inputs = tf.random_uniform(
            (1, crop_size[0], crop_size[1], 3))
        outputs_to_scales_to_logits = model.multi_scale_logits(
            inputs,
            model_options,
            image_pyramid=[1.0])

        sess.run(tf.global_variables_initializer())
        outputs_to_scales_to_logits = sess.run(outputs_to_scales_to_logits)
        output_node_names = 'ResizeBilinear_1'

        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )

        transforms = [
            'strip_unused_nodes(type=float,shape=\"1,33,33,3\")',
            'remove_nodes(op=Identity, op=CheckNumerics)',
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms',
            'fold_old_batch_norms'
        ]
        # x.set_shape([1, 224, 224, 3])
        from tensorflow.tools.graph_transforms import TransformGraph
        with tf.Graph().as_default() as graph:
            constant_graph = TransformGraph(
                constant_graph,
                ['MobilenetV2/MobilenetV2/input:0'],
                ['ResizeBilinear_1:0'],
                transforms
            )
            tf.import_graph_def(constant_graph, name="")

            for op in graph.get_operations():
                    print(op.name)

            x = graph.get_tensor_by_name('MobilenetV2/MobilenetV2/input:0')
            y = graph.get_tensor_by_name('ResizeBilinear_1:0')
            with tf.Session(graph=graph) as sess:
                converter = tf.contrib.lite.TFLiteConverter.from_session(sess, [x], [y])
                tflite_model = converter.convert()
                open("converted_model.tflite", "wb").write(tflite_model)
                
                # Load TFLite model and allocate tensors.
                interpreter = tf.contrib.lite.Interpreter(model_content=tflite_model)
                interpreter.allocate_tensors()

                # Get input and output tensors.
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                import numpy as np
                input_data = np.random.rand(1, 33, 33, 3).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)

                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                print(output_data)

        # Check computed results for each output type.
        for output in outputs_to_num_classes:
          scales_to_logits = outputs_to_scales_to_logits[output]
          # Expect only one output.
          self.assertEqual(len(scales_to_logits), 1)
          for logits in scales_to_logits.values():
            self.assertTrue(logits.any())

#   def testBuildDeepLabWithDensePredictionCell(self):
#     batch_size = 1
#     crop_size = [33, 33]
#     outputs_to_num_classes = {'semantic': 2}
#     expected_endpoints = ['merged_logits']
#     dense_prediction_cell_config = [
#       {'kernel': 3, 'rate': [1, 6], 'op': 'conv', 'input': -1},
#       {'kernel': 3, 'rate': [18, 15], 'op': 'conv', 'input': 0},
#     ]
#     model_options = common.ModelOptions(
#         outputs_to_num_classes,
#         crop_size,
#         output_stride=16)._replace(
#         aspp_with_batch_norm=True,
#         model_variant='mobilenet_v2',
#         dense_prediction_cell_config=dense_prediction_cell_config)
#     g = tf.Graph()
#     with g.as_default():
#       with self.test_session(graph=g):
#         inputs = tf.random_uniform(
#             (batch_size, crop_size[0], crop_size[1], 3))
#         outputs_to_scales_to_model_results = model.multi_scale_logits(
#             inputs,
#             model_options,
#             image_pyramid=[1.0])
#         for output in outputs_to_num_classes:
#           scales_to_model_results = outputs_to_scales_to_model_results[output]
#           self.assertListEqual(scales_to_model_results.keys(),
#                                expected_endpoints)
#           self.assertEqual(len(scales_to_model_results), 1)


if __name__ == '__main__':
  tf.test.main()
