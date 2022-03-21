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
from npu_bridge.npu_init import *
import nets.MobileFaceNet as MobileFaceNet
from losses.face_losses import cos_loss


# define the path of checkpoint
ckpt_path = "./checkpoint_npu/MobileFaceNet_best.ckpt"

def main(): 
    tf.reset_default_graph()

    phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), shape=None, name='phase_train')
    
    inputs = tf.placeholder(tf.float32, shape=[None, 112, 112, 3])
    prelogits, _ = MobileFaceNet.inference(inputs, phase_train=phase_train_placeholder, weight_decay=5e-5)
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10)
    
    flow = tf.cast(embeddings, tf.float16, 'the_outputs')
    saver = tf.train.Saver() 
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')    
        freeze_graph.freeze_graph(
		        input_graph='./pb_model/model.pb',   
		        input_saver='',
		        input_binary=False, 
		        input_checkpoint=ckpt_path,  
		        output_node_names='the_outputs',  
		        restore_op_name='save/restore_all',
		        filename_tensor_name='save/Const:0',
		        output_graph='./pb_model/MobileFaceNet.pb',   
		        clear_devices=False,
		        initializer_nodes='')
    print("done")

if __name__ == '__main__': 
    main()
    
    
