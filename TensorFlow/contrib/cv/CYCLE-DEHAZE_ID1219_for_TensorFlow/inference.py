# MIT License

# Copyright (c) 2018 Deniz Engin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ============================================================================
# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.png \
                       --output output_sample.png \
                       --image_size 256
"""
from npu_bridge.npu_init import *
import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import os
from model import CycleGAN
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', '', 'model path (.pb)')
tf.flags.DEFINE_string('input', 'input_sample.png', 'input image path (.png)')
tf.flags.DEFINE_string('output', 'output_sample.png', 'output image path (.png)')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')


def inference():
  graph = tf.Graph()

  with graph.as_default():
    with tf.gfile.FastGFile(FLAGS.input, 'rb') as f:
      image_data = f.read()
      input_image = tf.image.decode_jpeg(image_data, channels=3)
      input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
      input_image = utils.convert2float(input_image)
      input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])
    with tf.gfile.FastGFile(FLAGS.model, 'rb') as model_file:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(model_file.read())
    [output_image] = tf.import_graph_def(graph_def,
                          input_map={'input_image': input_image},
                          return_elements=['output_image:0'],
                          name='output')
  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer"
  config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  with tf.Session(graph=graph, config=npu_config_proto(config_proto=config)) as sess:
    generated = output_image.eval()
    with open(FLAGS.output, 'wb') as f:
      f.write(generated)

def main(unused_argv):
  inference()

if __name__ == '__main__':
  tf.app.run()

