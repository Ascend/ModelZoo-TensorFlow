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
import os
import time

import numpy as np
import tensorflow.compat.v1 as tf
from absl import flags
from tensorflow_core.python.client import timeline

FLAGS = flags.FLAGS

OUT_TYPES = {
    "FP32": np.float32,
    "FP16": np.float16,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
}


def pb_predict():
    data_dir = FLAGS.data_dir
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.task_name + "_CPU")
    predict_batch_size = FLAGS.predict_batch_size
    pb_model_file = FLAGS.pb_model_file
    max_seq_length = FLAGS.max_seq_length
    in_nodes = FLAGS.in_nodes.strip("\"").split(";")
    out_nodes = FLAGS.out_nodes.strip("\"").split(";")
    output_type = FLAGS.output_type

    input_file_dict = {}
    for idx, in_node in enumerate(in_nodes):
        input_idx = []
        node_name = in_node.split(":")[0]
        for root, dirs, files in os.walk(os.path.join(data_dir, "%s_%s" % (node_name, predict_batch_size))):
            for file in files:
                if file.endswith(".bin"):
                    input_idx.append(os.path.join(root, file))
        input_idx.sort()
        input_file_dict[node_name] = input_idx

    if os.path.exists(output_dir):
        os.system("rm -rf %s" % output_dir)
    os.makedirs(output_dir)

    result_num = len(input_file_dict[list(input_file_dict)[0]])
    for i in range(result_num):
        tf.reset_default_graph()
        config = tf.ConfigProto()

        with tf.io.gfile.GFile(os.path.join(pb_model_file), 'rb') as f:
            global_graph_def = tf.GraphDef.FromString(f.read())

        global_graph = tf.Graph()
        with global_graph.as_default():
            tf.import_graph_def(global_graph_def, name='')

        feed_dict = {}

        for key in input_file_dict.keys():
            input_name = global_graph.get_tensor_by_name("%s:0" % key)
            input_feed = np.fromfile(input_file_dict[key][i], dtype=np.int32).reshape(
                [predict_batch_size, max_seq_length])
            feed_dict[input_name] = input_feed

        outputs = []
        for out in out_nodes:
            outputs.append(global_graph.get_tensor_by_name(out))

        with tf.Session(graph=global_graph, config=config) as sess:
            run_options = None
            run_metadata = None

            predict = sess.run(outputs, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

            print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], "I",
                                            "Writing cpu predict result to file %s/%s" % (i + 1, result_num)))
            model_name = (FLAGS.pb_model_file.split('/')[-1]).split('.')[0]
            output_prefix = model_name + "_Idx"

            for out_idx, result in enumerate(predict):
                save_file = os.path.join(output_dir, '%s_%05d_output_%02d_000.bin' % (output_prefix, i, out_idx))
                np.array(result).astype(OUT_TYPES[output_type]).tofile(save_file)
