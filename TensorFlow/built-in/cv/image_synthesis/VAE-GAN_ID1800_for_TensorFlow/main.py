#
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
#
'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function
from npu_bridge.npu_init import *

import math
import os

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data

from progressbar import ETA, Bar, Percentage, ProgressBar

from vae import VAE
from gan import GAN

import time

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 128, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 128, "size of the hidden VAE unit")
flags.DEFINE_string("model", "gan", "gan or vae")

FLAGS = flags.FLAGS

if __name__ == "__main__":
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    mnist = input_data.read_data_sets(data_directory, one_hot=True)

    assert FLAGS.model in ['vae', 'gan']
    if FLAGS.model == 'vae':
        model = VAE(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)
    elif FLAGS.model == 'gan':
        model = GAN(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)

    for epoch in range(FLAGS.max_epoch):
        training_loss = 0.0

        pbar = ProgressBar()
        for i in pbar(range(FLAGS.updates_per_epoch)):
            start_time = time.time()
            images, _ = mnist.train.next_batch(FLAGS.batch_size)
            loss_value = model.update_params(images)
            end_time = time.time() - start_time
            training_loss += loss_value

        training_loss = training_loss / \
            (FLAGS.updates_per_epoch * FLAGS.batch_size)

        print("Loss: %f Time: %.4f us" %(training_loss, end_time*1000000))

        model.generate_and_save_images(
            FLAGS.batch_size, FLAGS.working_directory)

