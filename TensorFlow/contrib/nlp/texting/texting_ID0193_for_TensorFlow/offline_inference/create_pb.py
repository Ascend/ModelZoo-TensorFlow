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
from models import GNN

ckpt_path = 'pre_trained/mr/model.ckpt'

def main():
    tf.reset_default_graph()

    support = tf.placeholder(tf.float32, shape=(None, 46, 46), name='support')
    features = tf.placeholder(tf.float32, shape=(None, 46, 300), name='features')
    mask = tf.placeholder(tf.float32, shape=(None, 46, 1), name='mask')
    labels = tf.placeholder(tf.float32, shape=(None, 2), name='labels')
    dropout = 0.

    placeholders = {
        'support': support, 
        'features': features,
        'mask': mask,
        'labels': labels,
        'dropout': dropout
    }

    model = GNN(placeholders, input_dim=300, logging=False)
    logits = model.predict()
    predict_class = tf.argmax(logits, 1, name='output')
   
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
            output_graph='save/texting.pb',
            clear_devices=False,
            initializer_nodes=''
        )
    print('done')

if __name__ == '__main__':
    main()
