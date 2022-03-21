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
# Author: Bichen Wu (bichen@berkeley.edu) 03/07/2017

import os

os.chdir('/home/ma-user/modelarts/user-job-dir/code/')

from npu_bridge.estimator import npu_ops
import tensorflow as tf
from tensorflow.python.framework import graph_util
from nets.squeezeSeg import SqueezeSeg
from config.kitti_squeezeSeg_config import kitti_squeezeSeg_config

mc = kitti_squeezeSeg_config()
mc.LOAD_PRETRAINED_MODEL = False
mc.BATCH_SIZE = 1
mc.KEEP_PROB = 1.0
ph_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_path', "./inference/model.ckpt-25500",
                           """Directory of checkpoint. """)
tf.app.flags.DEFINE_string('frozen_pb_path', "/home/ma-user/modelarts/outputs/train_url_0/frozen_model.pb",
                           """Directory where to write frozen_pb""")


def frozen_pb():
    tf.reset_default_graph()
    # set inputs node
    ph_lidar_input = tf.placeholder(tf.float32, [1, 64, 512, 5], name="Placeholder")
    ph_lidar_mask = tf.placeholder(tf.float32, [1, 64, 512, 1], name="Placeholder_1")
    ph_label = tf.placeholder(tf.int32, [1, 64, 512], name="Placeholder_2")
    ph_loss_weight = tf.placeholder(tf.float32, [1, 64, 512], name="Placeholder_3")
    inputs_list = [ph_lidar_input, ph_lidar_mask, ph_label, ph_loss_weight]
    model = SqueezeSeg(mc, inputs_list)  # 神经网络的输出
    prob = tf.multiply(tf.nn.softmax(model.output_prob, dim=-1), ph_lidar_mask)
    output = tf.argmax(prob, axis=3, output_type=tf.int32, name='output')
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_graph = FLAGS.frozen_pb_path
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.checkpoint_path)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=["output"])
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    print("done")


def main(argv=None):
    frozen_pb()


if __name__ == '__main__':
    tf.app.run()
