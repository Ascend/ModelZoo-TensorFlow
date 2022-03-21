# -*-coding: utf-8 -*-
#
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#
from tensorflow.python.tools import freeze_graph
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
import os
from tensorflow.python.framework import tensor_util
from google.protobuf import text_format
from tensorflow.python.platform import gfile


resize_height = 299  # 指定图片高度
resize_width = 299  # 指定图片宽度
depths = 3
flags = tf.app.flags

flags.DEFINE_string("ckpt_file", None, "checkpoint file")
flags.DEFINE_string("pb_file", None, "pb_dir")
FLAGS = flags.FLAGS

def freeze_graph_new(input_checkpoint, output_graph):
    output_node_names = "pose_and_explainability_prediction/pose_exp_net/pose/mul"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices = True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess = sess,
            input_graph_def = input_graph_def,
            output_node_names = output_node_names.split(",")
        )

        with tf.gfile.GFile(output_graph, "wb") as f:  # save model
            f.write(output_graph_def.SerializeToString())  # sequence output
        print("%d ops in the final graph." % len(output_graph_def.node))
        tf.train.write_graph(output_graph_def, "./pb/", "out.txt", as_text=True)


if __name__ == '__main__':
    freeze_graph_new(FLAGS.ckpt_file, FLAGS.pb_file)
