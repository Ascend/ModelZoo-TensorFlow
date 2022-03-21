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
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './MNISTS-data', 'data dir')

# Get input
from tensorflow.examples.tutorials.mnist import input_data
# mnist_data = '/home/dataset/mnist'
mnist_data = FLAGS.data_dir
def mnist_data_iteratior():
    mnist = input_data.read_data_sets(mnist_data, one_hot=True)#./data/mnist
    def iterator(hparams, num_batches):
        for _ in range(num_batches):
            yield mnist.train.next_batch(hparams.batch_size)
    return iterator
