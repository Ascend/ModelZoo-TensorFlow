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

import layers as L
import cnn

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_path', './model.ckpt-60000', "the path of ckpt")



def logit(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):
    return cnn.logit(x, is_training=is_training,
                     update_batch_stats=update_batch_stats,
                     stochastic=stochastic,
                     seed=seed)[0]


def forward(x, is_training=True, update_batch_stats=True, seed=1234):
    if is_training:
        return logit(x, is_training=True,
                     update_batch_stats=update_batch_stats,
                     stochastic=True, seed=seed)
    else:
        return logit(x, is_training=False,
                     update_batch_stats=update_batch_stats,
                     stochastic=False, seed=seed)


def main(): 
    with tf.variable_scope("CNN",reuse=tf.AUTO_REUSE) as scope:
        inputs = tf.placeholder(tf.float32, shape=[1, 32, 32, 3], name="image")
        logit1 = forward(inputs, is_training=False, update_batch_stats=True)
    # ???????????????????????????
        predict_class = tf.argmax(logit1, 1,output_type=tf.int32, name="output")
    with tf.Session() as sess:
        
        #???????????????./pb_model??????????????????model.pb??????
        # model.pb???????????????input_graph??????????????????freeze_graph??????
            tf.train.write_graph(sess.graph_def, './pb_model', 'model_cifar.pb')    # ??????write_graph??????????????????
        
            freeze_graph.freeze_graph(
                    input_graph='./pb_model/model.pb',   # ??????write_graph?????????????????????
                    input_saver='',
                    input_binary=False, 
                    input_checkpoint=FLAGS.model_path,  # ?????????????????????checkpoint??????
                    output_node_names='CNN/output',  # ????????????????????????????????????????????????
                    restore_op_name='save/restore_all',
                    filename_tensor_name='save/Const:0',
                    output_graph='./pb_model/cifar.pb',   # ??????????????????????????????????????????
                    clear_devices=False,
                    initializer_nodes='')
    print("done")

if __name__ == '__main__': 
    main()