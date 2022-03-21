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
from tensorflow.python.platform import gfile
import tensorflow as tf 
import numpy as np
import os 
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pb_model', type=str, default='./offline_infer/flownet2.pb')
parser.add_argument('--output', type=str, default='./offline_infer/Bin/outputs_pb')
parser.add_argument('--img_dir', type=str, default='./offline_infer/Bin/image')
parser.add_argument('--gt_dir', type=str, default='./offline_infer/Bin/gt')
args = parser.parse_args()
sess = tf.Session()
with gfile.FastGFile(args.pb_model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())
 
output_shape = [1, 448, 1024, 2]
img_shape = [1, 2, 448, 1024, 3]
#img2_shape = [1, 448, 1024, 3]
output_dir = args.output
img_dir = args.img_dir
gt_dir = args.gt_dir
os.makedirs(output_dir, exist_ok=True)

names_images = sorted(os.listdir(img_dir))
cost = []
for name in names_images:
    filename = os.path.join(img_dir, name)
    mask_filename = os.path.join(gt_dir, name)

    test_img = np.fromfile(filename, dtype=np.float32).reshape([1, 2, 448, 1024, 3])
    start = datetime.datetime.now()
    output = sess.run('output:0', feed_dict={'input_a:0': test_img})
    end = datetime.datetime.now()
    deltatime = (end - start).total_seconds() * 1000
    cost.append(deltatime)
    # test_mask = np.fromfile(mask_filename, dtype=np.float32).reshape([1, 256, 256])

    # output = sess.run('output:0', feed_dict={'input_img:0': test_img, 'input_mask:0': test_mask})
    output.tofile(os.path.join(output_dir, name))

cost = np.array(cost)
c1 = np.mean(cost)
c2 = np.mean(cost[1:])
print('mean time cost: {}'.format(c1))
print('mean time cost without first batch: {}'.format(c2))
