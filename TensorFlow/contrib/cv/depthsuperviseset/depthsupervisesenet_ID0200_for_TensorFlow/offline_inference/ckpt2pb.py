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

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import util.FaceMapNet_BAS as FaceMapNet_BAS
import sys, argparse

def main(args):
    tf.reset_default_graph()
    color_batch_p = tf.compat.v1.placeholder(tf.float32, shape=(None, 256, 256, 3), name='color')
    depth_label_batch_p = tf.compat.v1.placeholder(tf.float32, shape=(None, 32, 32, 3), name='depth_label')
    label_batch_P = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='label')
    domain_batch_p = tf.compat.v1.placeholder(tf.string, shape=(None,), name='domain')
    model_list = FaceMapNet_BAS.build_Multi_Adversarial_Loss(color_batch_p, depth_label_batch_p,
        label_batch_P, domain_batch_p, 2, [0.1, 0, 0.1], depth_size=32, isTraining=False)

    graph = tf.get_default_graph()
    op = graph.get_operations()
    for i, m in enumerate(op):
        try:
            print(m.values()[0])
        except:
            break

    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, './offline_inference/pbModel', 'model.pb')

        freeze_graph.freeze_graph(
              input_graph='./offline_inference/pbModel/model.pb',
              input_saver='',
              input_binary=False,
              input_checkpoint=args.ckpt_path,
              output_node_names='DE_color/conv4_3/Conv2D',  ### cross_entropy/cross_entropy & DE_color/fc_logit/dense/MatMul
              restore_op_name='save/restore_all',
              filename_tensor_name='save/Const:0',
              output_graph='./offline_inference/pbModel/depthNet_tf.pb',
              clear_devices=False,
              initializer_nodes='')
    print("done")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default='./offline_inference/ckptModel/model-001.ckpt-42')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
