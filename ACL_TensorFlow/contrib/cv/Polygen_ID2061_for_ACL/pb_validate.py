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

import os
import time

import tensorflow as tf
import numpy as np


def vertex_predict(pb_path, x):
    with tf.Graph().as_default() as g:
        output_graph_def = tf.compat.v1.GraphDef()
        init = tf.compat.v1.global_variables_initializer()
        with open(pb_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            tf.graph_util.import_graph_def(output_graph_def)

        layers = [op.name for op in g.get_operations()]
        # print(layers)

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            input_x = sess.graph.get_tensor_by_name("import/v_input:0")
            output = sess.graph.get_tensor_by_name("import/v_output:0")

            # you can use this directly
            feed_dict = {
                input_x: x,
            }
            ret = sess.run(output, feed_dict=feed_dict)

        return ret


def face_predict(pb_path, vertex, mask):
    with tf.Graph().as_default() as g:
        output_graph_def = tf.compat.v1.GraphDef()
        init = tf.compat.v1.global_variables_initializer()
        with open(pb_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            tf.graph_util.import_graph_def(output_graph_def)

        layers = [op.name for op in g.get_operations()]
        # print(layers)

        with tf.compat.v1.Session() as sess:
            sess.run(init)
            input1 = sess.graph.get_tensor_by_name("import/f_input:0")
            input2 = sess.graph.get_tensor_by_name("import/f_mask:0")
            output = sess.graph.get_tensor_by_name("import/f_output:0")

            # you can use this directly
            feed_dict = {
                input1: vertex,
                input2: mask
            }
            ret = sess.run(output, feed_dict=feed_dict)

        return ret


v_pb = 'pb/vertex_model.pb'
f_pb = 'pb/face_model.pb'
label = np.fromfile("bin/label.bin", dtype="int32")
print(label)
start1 = time.time()
res = vertex_predict(v_pb, label)
end1 = time.time()
print(end1 - start1)
print(res)
print(res.shape)
res.tofile("out/pb/vertex_model.bin")

start2 = time.time()
f = face_predict(f_pb, res[..., :3], np.squeeze(res[..., 3:], axis=2))
end2 = time.time()
print(end2 - start2)
print(f)
print(f.shape)
f.tofile("out/pb/face_model.bin")
