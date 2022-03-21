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
import tensorflow as tf
from test_preprocess import pkl_load
import numpy as np


def inf_acc(pb_path):
    '''
    :param pb_path: [str] pb_model path
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # input tensor name
            seq_inputs = sess.graph.get_tensor_by_name("seq_inputs:0")
            item_inputs = sess.graph.get_tensor_by_name("item_inputs:0")

            # output tensor name
            output_tensor_name = sess.graph.get_tensor_by_name("out:0")

            pkl = pkl_load('./raw_data/cache.pkl')
            test_X, test_y = pkl['test_X'], pkl['test_y']
            x_dense_inputs, x_sparse_inputs, x_seq_inputs, x_item_inputs = test_X

            # model test
            out = sess.run(output_tensor_name, feed_dict={
                seq_inputs: x_seq_inputs,
                item_inputs: x_item_inputs})
            out = np.round(out).reshape(len(out), )
            acc = np.sum(np.equal(out, test_y)) / len(out)
            print(">>>>> ", "%d test units ,Accuracy: %.6f" % (len(out), acc))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pb_path', default='')
    args = parser.parse_args()
    inf_acc(pb_path=args.pb_path)
