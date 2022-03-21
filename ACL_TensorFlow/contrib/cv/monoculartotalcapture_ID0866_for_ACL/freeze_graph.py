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
from nets.CPM import CPM
ckpt_path = "./model/model-400000"

def main(): 
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[1, 368, 368, 3], name="input")
    net = CPM(out_chan=21, crop_size=368, withPAF=True, PAFdim=3, numPAF=23, numStage=5)
    predicted_scoremaps, _, predicted_PAFs = net.inference(inputs, train=False)
    predicted_scoremaps = tf.stack(predicted_scoremaps, axis=0, name='output0')
    predicted_PAFs = tf.stack(predicted_PAFs, axis=0, name='output2')
    with tf.Session() as sess:
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')    
        freeze_graph.freeze_graph(
		        input_graph='./pb_model/model.pb',  
		        input_saver='',
		        input_binary=False, 
		        input_checkpoint=ckpt_path,  # 传入训练生成的checkpoint文件
		        output_node_names='output0,CPM/out_11,output2', 
		        restore_op_name='save/restore_all',
		        filename_tensor_name='save/Const:0',
		        output_graph='./pb_model/monoculartotalcapture.pb',   
		        clear_devices=False,
		        initializer_nodes='')
    print("done")

if __name__ == '__main__': 
    main()

