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
# ==============================================================================
import tensorflow as tf
from core.yolov3 import YOLOV3
from core.yolov4 import YOLOV4
from .darknet_weights_to_ckpt import load_weights


if __name__ == '__main__':
    input_size = 416
    darknet_weights = 'model_data/yolov4.weights'
    pb_file = './yolov4.pb'
    output_node_names = ["input/input_data", "pred_sbbox/concat_2", "pred_mbbox/concat_2", "pred_lbbox/concat_2"]

    # you need to reset graph first.
    tf.reset_default_graph()
    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, name='input_data')

    model = YOLOV4(input_data, trainable=False)
    load_ops = load_weights(tf.global_variables(), darknet_weights)

    with tf.Session() as sess:
        sess.run(load_ops)
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
            #tf.get_default_graph().as_graph_def(),
            sess.graph.as_graph_def(), output_node_names=output_node_names)

        with tf.gfile.GFile(pb_file, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("{} ops written to {}.".format(len(output_graph_def.node), pb_file))
