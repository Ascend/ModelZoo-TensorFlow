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
from tensorflow.python.platform import gfile

FLAGS = flags.FLAGS


def convert_pbtxt():
    pbtxt_model_file = FLAGS.pb_model_file.split('/')[-1] + 'txt'
    with gfile.FastGFile(FLAGS.pb_model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, FLAGS.output_dir, pbtxt_model_file, as_text=True)
    print("%s - %s - [XNLP]: %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
                                    "I", "Finish to convert pb to pbtxt model: %s" % pbtxt_model_file))
