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
import sys
from tensorflow.python.tools import freeze_graph
from npu_bridge.estimator import npu_ops 
from functions.model_fns import Model


ckpt_path = '/home/sunshk/food_ckpt/model.ckpt-220038'

def main(): 
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name="input")
    
    model = Model(50, num_classes=101,
                      resnet_version=2,
                      zero_gamma=False,
                      use_se_block=False,
                      use_sk_block=True,
                      no_downsample=False,
                      anti_alias_filter_size=3,
                      anti_alias_type='sconv',
                      bn_momentum=0.966,
                      embedding_size=0,
                      pool_type='gap',
                      bl_alpha=2,
                      bl_beta=4,
                      dtype=tf.float32,
                      loss_type='softmax')
    
    logits = model(inputs, False, False,
                     use_resnet_d=False, keep_prob=1.0)
    
    predict_class = tf.argmax(logits, axis=1, output_type=tf.int32, name="output")
    
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')    
        freeze_graph.freeze_graph(
            input_graph='./pb_model/model.pb',   
            input_binary=False, 
            input_checkpoint=ckpt_path,  
            output_node_names='output',  
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0',
            output_graph='./pb_model/npu.pb',   # name of .pb
            clear_devices=False,
            initializer_nodes='')
    print("done")

main()