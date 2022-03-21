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
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

def predict(pb_path, x, y):

  with tf.Graph().as_default() as g:
    output_graph_def = tf.compat.v1.GraphDef()
    init = tf.compat.v1.global_variables_initializer()
    """
    load pb model
    """
    with open(pb_path, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        tf.graph_util.import_graph_def(output_graph_def) #name是必须的
    
    layers = [op.name for op in g.get_operations()]
  
    """
    enter a text and predict
    """
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        input_z = sess.graph.get_tensor_by_name(
            "import/x1:0")
        input_label = sess.graph.get_tensor_by_name(
            "import/x2:0")
        output = "import/Identity:0"
    
        # you can use this directly
        feed_dict = {
            input_z: x,
            input_label: y
        }
        y_pred_cls = sess.run(output, feed_dict=feed_dict)
    return y_pred_cls
  
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_bin_dir', type=str, help='dir where to input bin.',default= "./Bin")
    parser.add_argument('--input_pb_path', type=str, help='path where to load pb file.',default= "./models/results/pb")
    parser.add_argument('--output_pb_generated_img', type=str, help='dir where to output generated images.',default= "./models/results/generated_images")
    return parser.parse_args(argv)

def main(args):
    x1 = np.fromfile(args.input_bin_dir + "/fake_image.bin", dtype='float32')
    x1 = x1.reshape([256,100])
    x2 = np.fromfile(args.input_bin_dir + "/label.bin", dtype='int32')
    x2 = x2.reshape([256])
    # Save the results of pb generated pictures
    pb_out = predict(args.input_pb_path + '/bigbigan.pb', x1, x2)
    pb_out = pb_out.astype("float32")
    pb_out.tofile(args.output_pb_generated_img + "/pb_out.bin")
    
if __name__ == "__main__" :
    main(parse_arguments(sys.argv[1:]))