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
from npu_bridge.npu_init import *
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import os
from Train.config import args
from help_modelarts import modelarts_result2obs

from Train.resnet import ResNet
from Train.config import config as net_config

INIT_CKPT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpoint65')
ckpt_path = os.path.join(INIT_CKPT_DIR, 'model.ckpt-65000')

def main():
    print("start ckpt To pb")
    print("ckpt_path")
    tf.reset_default_graph()
    img_ph = tf.placeholder(tf.float32, shape=[1, 300, 300, 3], name="input")
    dataset_num_classes = 21

    net = ResNet
    depth = 50
    net = net(config=net_config, depth=depth, training=False)

    net.create_trunk(img_ph)

    if args.detect:
        net.create_multibox_head(dataset_num_classes)
        confidence = net.outputs['confidence']
        location = net.outputs['location']
    else:
        location, confidence = None, None

    if args.segment:
        net.create_segmentation_head(dataset_num_classes)
        seg_logits = net.outputs['segmentation']
    else:
        seg_logits = None

    print("confidence = ", confidence)
    print("location = ", location)
    print("seg_logits = ", seg_logits)

    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, args.result_dir, 'model.pb')
        modelarts_result2obs(args)
        freeze_graph.freeze_graph(
            input_graph=os.path.join(args.result_dir, 'model.pb'),
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names="concat_1, concat_2, ssd_2/Conv_7/BiasAdd",  # graph outputs node
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph=os.path.join(args.result_dir, 'bliznet_tf_310.pb'),  # graph outputs name
            clear_devices=False,
            initializer_nodes="")
    print("done")

    modelarts_result2obs(args)

if __name__ == '__main__':
    main()