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
"""
Freeze Lanenet model into frozen pb file
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def convert_ckpt_into_pb_file(ckpt_file_path, pb_file_path):
    """

    :param ckpt_file_path:
    :param pb_file_path:
    :return:
    """
    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(ckpt_file_path + '.meta', clear_devices=True)
        input_graph_def = tf.get_default_graph().as_graph_def()

        binary_seg_node = 'lanenet/binary_seg_out'
        instance_seg_node = 'lanenet/instance_seg_out'

        saver.restore(sess, ckpt_file_path)  # 恢复图并得到数据
        output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=['input_tensor', binary_seg_node, instance_seg_node])

        with tf.gfile.GFile(pb_file_path, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    """
    test code
    """

    ckpt_path = './eval_ckpt/eval.ckpt'
    pb_save_path = './pretrained_model/eval.pb'

    convert_ckpt_into_pb_file(
        ckpt_file_path=ckpt_path,
        pb_file_path=pb_save_path
    )
