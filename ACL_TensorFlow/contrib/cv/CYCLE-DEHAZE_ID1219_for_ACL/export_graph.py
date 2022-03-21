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
""" Freeze variables and convert 2 generator networks to 2 GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:
python export_graph.py --checkpoint_dir checkpoints/20170424-1152 \
                       --XtoY_model apple2orange.pb \
                       --YtoX_model orange2apple.pb \
                       --image_size 256
"""
from npu_bridge.npu_init import *

import tensorflow as tf
import os
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from model import CycleGAN
import utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', '', 'checkpoints directory path')
tf.flags.DEFINE_string('model_dir', '', 'pb directory path')
tf.flags.DEFINE_string('XtoY_model', 'apple2orange.pb',
                       'XtoY model name, default: apple2orange.pb')
tf.flags.DEFINE_string('YtoX_model', 'orange2apple.pb',
                       'YtoX model name, default: orange2apple.pb')
tf.flags.DEFINE_integer('image_size1', '256', 'image size, default: 256')
tf.flags.DEFINE_integer('image_size2', '256', 'image size, default: 256')
tf.flags.DEFINE_integer(
    'ngf', 64, 'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string(
    'norm', 'instance',
    '[instance, batch] use instance norm or batch norm, default: instance')


def export_graph(model_name, XtoY=True):
    graph = tf.Graph()

    with graph.as_default():
        cycle_gan = CycleGAN(ngf=FLAGS.ngf,
                             norm=FLAGS.norm,
                             image_size1=FLAGS.image_size1,
                             image_size2=FLAGS.image_size2)

        input_image = tf.placeholder(
            tf.float32,
            shape=[FLAGS.image_size1, FLAGS.image_size2, 3],
            name='input_image')
        cycle_gan.model()
        if XtoY:
            output_image = cycle_gan.G.sample(tf.expand_dims(input_image, 0))
        else:
            output_image = cycle_gan.F.sample(tf.expand_dims(input_image, 0))

        output_image = tf.identity(output_image, name='output_image')
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(
        "allow_mix_precision")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    with tf.Session(graph=graph,
                    config=npu_config_proto(config_proto=config)) as sess:
        sess.run(tf.global_variables_initializer())
        latest_ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        restore_saver.restore(sess, latest_ckpt)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [output_image.op.name])

        tf.train.write_graph(output_graph_def,
                             FLAGS.model_dir,
                             model_name,
                             as_text=False)


def main():
    print('Export XtoY model...')
    export_graph(FLAGS.XtoY_model, XtoY=True)
    print('Export YtoX model...')
    export_graph(FLAGS.YtoX_model, XtoY=False)


if __name__ == '__main__':
    tf.app.run()
