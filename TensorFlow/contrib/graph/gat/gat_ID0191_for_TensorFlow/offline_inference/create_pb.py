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
from tensorflow.python.tools import freeze_graph
from models import GAT

ckpt_path = 'pre_trained/cora/mod_cora.ckpt'

def main():
    tf.reset_default_graph()

    nb_nodes = 2708
    ft_size = 1433
    nb_classes = 7
    ftr_in = tf.placeholder(dtype=tf.float32, shape=(None, nb_nodes, ft_size), name='ftr_in')
    bias_in = tf.placeholder(dtype=tf.float32, shape=(None, nb_nodes, nb_nodes), name='bias_in')
    lbl_in = tf.placeholder(dtype=tf.int32, shape=(None, nb_nodes, nb_classes), name='lbl_in')
    msk_in = tf.placeholder(dtype=tf.int32, shape=(None, nb_nodes), name='msk_in')
    attn_drop = 0.0 #tf.placeholder(dtype=tf.float32, shape=())
    ffd_drop = 0.0 #tf.placeholder(dtype=tf.float32, shape=())
    is_train = False #tf.placeholder(dtype=tf.bool, shape=())

    logits = GAT.inference(ftr_in, nb_classes, nb_nodes, is_train,
                             attn_drop, ffd_drop,
                             bias_mat=bias_in,
                             hid_units=[8], n_heads=[8,1],
                             residual=False, activation=tf.nn.elu)
    log_resh = tf.reshape(logits, [-1, nb_classes])
    correct_prediction = tf.argmax(log_resh, 1, name='output')

    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, 'save', 'model.pb')
        freeze_graph.freeze_graph(
            input_graph='save/model.pb',
            input_saver='',
            input_binary=False,
            input_checkpoint=ckpt_path,
            output_node_names='output',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='save/gat.pb',
            clear_devices=False,
            initializer_nodes=''
        )
    print('done')

if __name__ == '__main__':
    main()