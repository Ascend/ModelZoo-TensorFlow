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
import numpy as np
import sys
import os

config = tf.ConfigProto()

label_path = sys.argv[1]
npu_predict = sys.argv[2]

final_output = tf.placeholder(shape=[None,2],dtype=tf.float32,name='final_output')
target_ph =tf.placeholder(shape=[None,2],dtype=tf.float32,name='target_ph')

y_hat = tf.add(final_output,0.00000001)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y_hat), target_ph), tf.float32))

acc_list = []
init = tf.global_variables_initializer()
with tf.Session(config=config) as sess:
    file_list = os.listdir(label_path)
    file_list.sort()
    for file in file_list:
        if file.endswith('.bin'):
            label = np.fromfile(os.path.join(label_path,file), dtype='float32')
            label = label.reshape(int(label.shape[0]/2),2)
            npu_res = np.fromfile(os.path.join(npu_predict,"davinci_{}_output0.bin".format(file.split('.bin')[0])), dtype='float32')
            npu_res = npu_res.reshape(int(npu_res.shape[0]/2),2)
            acc = sess.run(accuracy,feed_dict={final_output:npu_res,target_ph:label})
            acc_list.append(acc.astype('float32'))
    print("Mean Accuracy: {}".format(sum(acc_list)/len(acc_list)))
