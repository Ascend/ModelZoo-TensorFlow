# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
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
# Lint as: python2, python3


import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from architecture import shufflenet_v2

# checkpoint path
ckpt_path = "./model/model.ckpt"

def main():
    tf.reset_default_graph()
    # Input Node
    features = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
    # Inference graph
    is_training = False
    logits = shufflenet_v2(
        features, is_training,
        num_classes=1000,
        depth_multiplier='0.5'
    )
    # Output Node
    logits = tf.nn.softmax(logits, axis=1, name="logits")
    predict_class = tf.argmax(logits, axis=1, output_type=tf.int32, name="output")
    with tf.Session() as sess:
        
        tf.train.write_graph(sess.graph_def, './test_pb_model', 'model.pb')    
        freeze_graph.freeze_graph(input_graph='./pb_model/model.pb',   
                                  input_saver='',
                                  input_binary=False,
                                  input_checkpoint=ckpt_path,  
                                  output_node_names='output',  
                                  restore_op_name='save/restore_all',
                                  filename_tensor_name='save/Const:0',
                                  output_graph='./test_pb_model/shufflenetv2.pb',   
                                  clear_devices=False,
                                  initializer_nodes=''
                                  )
    print("done")

if __name__ == '__main__':
    main()
