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

"""
    Here, we show the method to preprocess the input data as *.bin file.
"""
from tensorflow.python.platform import gfile
import tensorflow as tf
from PIL import Image
import numpy as np

# from modelnew import prototypical, euclidean_distance
from data import Data

n_way = 60
n_shot = 5
n_query = 5
n_examples = 20
im_width = 28
im_height = 28
channels = 1
h_dim = 64
z_dim = 64

n_test_episodes = 1000
n_test_way = 20
n_test_shot = 5
n_test_query = 15


# testing data
test_data = Data(n_examples=n_examples, im_height=im_height, im_width=im_width, datatype='test.txt')
print(test_data.shape)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epi in range(n_test_episodes):
        #data2bin
        epi_classes_test = np.random.permutation(test_data.n_classes)[:n_test_way]
        support_test = np.zeros([n_test_way, n_test_shot, im_height, im_width], dtype=np.float32)
        query_test = np.zeros([n_test_way, n_test_query, im_height, im_width], dtype=np.float32)
        for i, epi_cls in enumerate(epi_classes_test):
            selected_test = np.random.permutation(n_examples)[:n_test_shot + n_test_query]
            support_test[i] = test_data.dataset[epi_cls, selected_test[:n_test_shot]]
            query_test[i] = test_data.dataset[epi_cls, selected_test[n_test_shot:]]
        support_test = np.expand_dims(support_test, axis=-1)
        query_test = np.expand_dims(query_test, axis=-1)
        labels_test = np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.int64)
        # print(support_test.shape)
        # print(query_test.shape)
        # print(labels_test.shape)
        support_test.tofile("./input/inputx/input" + str(epi) + ".bin")
        query_test.tofile("./input/inputq/input" + str(epi) + ".bin")
        labels_test.tofile("./input/inputy/input" + str(epi) + ".bin")

    


