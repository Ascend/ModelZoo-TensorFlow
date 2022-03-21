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

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
# from npu_bridge.estimator import npu_ops
from architecture import shufflenet

# checkpoint path
ckpt_path = "./model/model.ckpt-1147669"

def main():
    tf.reset_default_graph()
    # Input Node
    features = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input")
    # Generate inference graph
    is_training = tf.constant(False, dtype=tf.bool)
    logits = shufflenet(
        features, is_training,
        num_classes=1000,
        groups=3,
        dropout=0.5,
        complexity_scale_factor=0.5
    )
    # Output Node
    logits = tf.nn.softmax(logits, axis=1, name="logits")
    predict_class = tf.argmax(logits, axis=1, output_type=tf.int32, name="output")
    with tf.Session() as sess:
        # Save model.pb to './pb_model'
        tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb')
        freeze_graph.freeze_graph(input_graph='./pb_model/model.pb',
                                  input_saver='',
                                  input_binary=False,
                                  input_checkpoint=ckpt_path,
                                  output_node_names='output',
                                  restore_op_name='save/restore_all',
                                  filename_tensor_name='save/Const:0',
                                  output_graph='./pb_model/shufflenetv1.pb',
                                  clear_devices=False,
                                  initializer_nodes=''
                                  )
    print("done")

if __name__ == '__main__':
    main()
