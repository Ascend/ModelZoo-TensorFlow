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

"""
Title: Image similarity estimation using a Siamese Network with a contrastive loss
Author: Mehdi
Date created: 2021/05/06
Last modified: 2021/05/06
Description: Similarity learning using a siamese network trained with a contrastive loss.
"""

"""
## Introduction

[Siamese Networks](https://en.wikipedia.org/wiki/Siamese_neural_network)
are neural networks which share weights between two or more sister networks,
each producing embedding vectors of its respective inputs.

In supervised similarity learning, the networks are then trained to maximize the
contrast (distance) between embeddings of inputs of different classes, while minimizing the distance between
embeddings of similar classes, resulting in embedding spaces that reflect
the class segmentation of the training inputs.
"""

import npu_device
import argparse
import ast
#===============================NPU Migration=========================================
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16,help='batch_size')
parser.add_argument('--epochs', type=int, default=10,help='batch_size')
parser.add_argument('--log_steps', type=int, default=3750, help='steps per log')
parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='the path to save over dump data')
parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval,
                    help='if or not over detection, default is False')
parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval,
                    help='data dump flag, default is False')
parser.add_argument('--data_dump_step', default="10",
                    help='data dump step, default is 10')
parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval,
                    help='use_mixlist flag, default is False')
parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval,
                    help='fusion_off flag, default is False')
parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
args = parser.parse_args()

def npu_config():
  if args.data_dump_flag:
    npu_device.global_options().dump_config.enable_dump = True
    npu_device.global_options().dump_config.dump_path = args.data_dump_path
    npu_device.global_options().dump_config.dump_step = args.data_dump_step
    npu_device.global_options().dump_config.dump_mode = "all"

  if args.over_dump:
      npu_device.global_options().dump_config.enable_dump_debug = True
      npu_device.global_options().dump_config.dump_path = args.over_dump_path
      npu_device.global_options().dump_config.dump_debug_mode = "all"

  if args.profiling:
      npu_device.global_options().profiling_config.enable_profiling = True
      profiling_options = '{"output":"' + args.profiling_dump_path + '", \
                          "training_trace":"on", \
                          "task_trace":"on", \
                          "aicpu":"on", \
                          "aic_metrics":"PipeUtilization",\
                          "fp_point":"", \
                          "bp_point":""}'
      npu_device.global_options().profiling_config.profiling_options = profiling_options
  npu_device.global_options().precision_mode = args.precision_mode
  if args.use_mixlist and args.precision_mode=='allow_mix_precision':
    npu_device.global_options().modify_mixlist=args.mixlist_file
  if args.fusion_off_flag:
    npu_device.global_options().fusion_switch_file=args.fusion_off_file
  npu_device.open().as_default()
#===============================NPU Migration=========================================
npu_config()

"""
## Setup
"""

import time
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import argparse

"""
## Hyperparameters
"""


epochs = args.epochs
batch_size = args.batch_size
margin = 1  # Margin for constrastive loss.

"""
## Load the MNIST dataset
"""

(x_train_val, y_train_val), (x_test, y_test) = keras.datasets.mnist.load_data()

# Change the data type to a floating point format
x_train_val = x_train_val.astype("float32")
x_test = x_test.astype("float32")


"""
## Define training and validation sets
"""

# Keep 50% of train_val  in validation set
x_train, x_val = x_train_val[:30000], x_train_val[30000:]
y_train, y_val = y_train_val[:30000], y_train_val[30000:]
del x_train_val, y_train_val


"""
## Create pairs of images

We will train the model to differentiate between digits of different classes. For
example, digit `0` needs to be differentiated from the rest of the
digits (`1` through `9`), digit `1` - from `0` and `2` through `9`, and so on.
To carry this out, we will select N random images from class A (for example,
for digit `0`) and pair them with N random images from another class B
(for example, for digit `1`). Then, we can repeat this process for all classes
of digits (until digit `9`). Once we have paired digit `0` with other digits,
we can repeat this process for the remaining classes for the rest of the digits
(from `1` until `9`).
"""


def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

    return np.array(pairs), np.array(labels).astype("float32")


# make train pairs
pairs_train, labels_train = make_pairs(x_train, y_train)

# make validation pairs
pairs_val, labels_val = make_pairs(x_val, y_val)

# make test pairs
pairs_test, labels_test = make_pairs(x_test, y_test)

"""
We get:

**pairs_train.shape = (60000, 2, 28, 28)**

- We have 60,000 pairs
- Each pair contains 2 images
- Each image has shape `(28, 28)`
"""

"""
Split the training pairs
"""

x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (60000, 28, 28)
x_train_2 = pairs_train[:, 1]

"""
Split the validation pairs
"""

x_val_1 = pairs_val[:, 0]  # x_val_1.shape = (60000, 28, 28)
x_val_2 = pairs_val[:, 1]

"""
Split the test pairs
"""

x_test_1 = pairs_test[:, 0]  # x_test_1.shape = (20000, 28, 28)
x_test_2 = pairs_test[:, 1]


"""
## Define the model

There are be two input layers, each leading to its own network, which
produces embeddings. A `Lambda` layer then merges them using an
[Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) and the
merged output is fed to the final network.
"""

# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


input = layers.Input((28, 28, 1))
x = tf.keras.layers.BatchNormalization()(input)
x = layers.Conv2D(4, (5, 5), activation="tanh")(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(16, (5, 5), activation="tanh")(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)

x = tf.keras.layers.BatchNormalization()(x)
x = layers.Dense(10, activation="tanh")(x)
embedding_network = keras.Model(input, x)


input_1 = layers.Input((28, 28, 1))
input_2 = layers.Input((28, 28, 1))

# As mentioned above, Siamese Network share weights between
# tower networks (sister networks). To allow this, we will use
# same embedding network for both tower networks.
tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)


"""
## Define the constrastive Loss
"""


def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

  Arguments:
      margin: Integer, defines the baseline for distance for which pairs
              should be classified as dissimilar. - (default is 1).

  Returns:
      'constrastive_loss' function with data ('margin') attached.
  """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.

      Returns:
          A tensor containing constrastive loss as floating point value.
      """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


"""
## Compile the model with the contrastive loss
"""

siamese.compile(loss=loss(margin=margin), optimizer="RMSprop", metrics=["accuracy"])

"""
## Add time history callbacks
"""

class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, log_steps, initial_step=0):
        self.batch_size = batch_size
        super(TimeHistory, self).__init__()
        self.steps_before_epoch = initial_step
        self.last_log_step = initial_step
        self.log_steps = log_steps
        self.steps_in_epoch = 0
        self.start_time = None
    
    @property
    def global_steps(self):
        """The current 1-indexed global step."""
        return self.steps_before_epoch + self.steps_in_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if not self.start_time:
            self.start_time = time.time()
        self.epoch_start = time.time()
    
    def on_batch_begin(self, batch, logs=None):
        if not self.start_time:
            self.start_time = time.time()
    
    def on_batch_end(self, batch, logs=None):
        self.steps_in_epoch = batch + 1
        steps_since_last_log = self.global_steps - self.last_log_step
        if steps_since_last_log >= self.log_steps:
            now = time.time()
            elapsed_time = now - self.start_time
            steps_per_second = steps_since_last_log / elapsed_time
            examples_per_second = steps_per_second * self.batch_size
            print(
                'TimeHistory: %.2f seconds, %.2f examples/second between steps %d '
                'and %d'%(elapsed_time, examples_per_second, self.last_log_step,
                self.global_steps),flush=True)
            self.last_log_step = self.global_steps
            self.start_time = None
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_run_time = time.time() - self.epoch_start
        self.steps_before_epoch += self.steps_in_epoch
        self.steps_in_epoch = 0

"""
## Train the model
"""

history = siamese.fit(
    [x_train_1, x_train_2],
    labels_train,
    validation_data=([x_val_1, x_val_2], labels_val),
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    callbacks=[TimeHistory(args.batch_size,args.log_steps)],
)
