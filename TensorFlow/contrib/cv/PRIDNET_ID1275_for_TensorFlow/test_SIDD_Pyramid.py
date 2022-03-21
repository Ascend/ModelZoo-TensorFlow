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

from __future__ import division
from scipy.io import loadmat
from scipy.io import savemat
import os, time
import tensorflow as tf
import numpy as np
from network import network 

flags=tf.flags
Flags=flags.FLAGS
flags.DEFINE_string('val_dir',"result",'train model result')
flags.DEFINE_string('checkpoint_dir','checkpoint','train ckpt dir')
flags.DEFINE_string('result_dir','mat','save mat dir')
val_dir = Flags.val_dir
checkpoint_dir = Flags.checkpoint_dir
result_dir = Flags.result_dir

#val_dir = '/home/test_user05/pridnet/ValidationNoisyBlocksRaw.mat'
#checkpoint_dir = '/home/test_user05/pridnet/checkpoint/model-3990.ckpt'
#result_dir = './res/'

mat = loadmat(val_dir)
val_img = mat['ValidationNoisyBlocksRaw']  # (40, 32, 256, 256)
val_img = val_img.reshape([1280, 256, 256])

ps = 256

ouput_blocks = [None] * 40 * 32

sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 1])
out_image = network(in_image)
saver = tf.train.Saver(max_to_keep=15)
sess.run(tf.global_variables_initializer())

print('loaded ' + checkpoint_dir)
saver.restore(sess, checkpoint_dir)

if not os.path.isdir(result_dir):
    print('-----------------------------no existing path')
    os.makedirs(result_dir)

for i in range(len(val_img)):
    each_block = val_img[i]  # (256, 256)
    each_block = np.expand_dims(np.expand_dims(each_block, axis=0), axis=3)

    st = time.time()
    output = sess.run(out_image, feed_dict={in_image: each_block})
    output = np.minimum(np.maximum(output, 0), 1)

    t_cost = time.time() - st
    ouput_blocks[i] = output
    print(ouput_blocks[i].shape)
    print('cleaning block %4d' % i)
    print('time_cost:', t_cost)
out_mat = np.squeeze(ouput_blocks)
out_mat = out_mat.reshape([40, 32, 256, 256])

savemat(result_dir + 'ValidationCleanBlocksRaw.mat', {'results': out_mat})