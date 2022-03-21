# coding=utf-8
# Copyright 2020 Huawei Technologies Co., Ltd
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

import datetime

import tensorflow.compat.v1 as tf
from absl import flags
from tensorflow.python.framework import graph_util

FLAGS = flags.FLAGS


def convert_pb(processor):
    if FLAGS.model_name.lower() == "transformer":
        print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                        "E", "Transformer network do not support freeze to pb"))
        exit(1)
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            processor.create_model()

            saver = tf.train.Saver()

            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
            tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['logits'])
            tmp_g = graph_util.remove_training_nodes(tmp_g)
            with tf.gfile.GFile(FLAGS.pb_model_file, mode='wb') as f:
                f.write(tmp_g.SerializeToString())
    print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                    "I", "Finish to convert checkpoints to pb model: %s" % FLAGS.pb_model_file))
