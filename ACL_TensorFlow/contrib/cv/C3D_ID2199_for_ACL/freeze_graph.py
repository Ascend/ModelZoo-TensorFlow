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
import C3D_model
from tensorflow.python.tools import freeze_graph
import numpy as np

model_path = '/home/test_user05/checkpoint/'
BATCH_SIZE = 60
NUM_CLASSES = 101
CROP_SZIE = 112
CHANNEL_NUM = 3
CLIP_LENGTH = 16
KEEP_PROB = 0.5

batch_clips = tf.placeholder(tf.float32, [BATCH_SIZE, CLIP_LENGTH, CROP_SZIE, CROP_SZIE, CHANNEL_NUM], name='batch_clips')
batch_labels = tf.placeholder(tf.int32, [BATCH_SIZE, NUM_CLASSES], name='batch_labels')
logits = C3D_model.C3D(batch_clips, NUM_CLASSES, KEEP_PROB)
output = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(batch_labels, 1)), np.float32), name="output")

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')
    freeze_graph.freeze_graph(
        input_graph='./pb_model/model.pb',
        input_saver='',
        input_binary=False,
        input_checkpoint=model_path + 'train.ckpt-39',
        output_node_names='output',
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        output_graph='./pb_model/c3d.pb',
        clear_devices=False,
        initializer_nodes='')
print('done')
