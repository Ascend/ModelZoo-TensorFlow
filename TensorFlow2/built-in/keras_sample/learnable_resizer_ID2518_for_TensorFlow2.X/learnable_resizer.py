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
Title: Learning to Resize in Computer Vision
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/04/30
Last modified: 2021/05/13
Description: How to optimally learn representations of images for a given resolution.
"""
"""
It is a common belief that if we constrain vision models to perceive things as humans do,
their performance can be improved. For example, in [this work](https://arxiv.org/abs/1811.12231),
Geirhos et al. showed that the vision models pre-trained on the ImageNet-1k dataset are
biased toward texture whereas human beings mostly use the shape descriptor to develop a
common perception. But does this belief always apply especially when it comes to improving
the performance of vision models?

It turns out it may not always be the case. When training vision models, it is common to
resize images to a lower dimension ((224 x 224), (299 x 299), etc.) to allow mini-batch
learning and also to keep up the compute limitations.  We generally make use of image
resizing methods like **bilinear interpolation** for this step and the resized images do
not lose much of their perceptual character to the human eyes. In
[Learning to Resize Images for Computer Vision Tasks](https://arxiv.org/abs/2103.09950v1), Talebi et al. show
that if we try to optimize the perceptual quality of the images for the vision models
rather than the human eyes, their performance can further be improved. They investigate
the following question:

**For a given image resolution and a model, how to best resize the given images?**

As shown in the paper, this idea helps to consistently improve the performance of the
common vision models (pre-trained on ImageNet-1k) like DenseNet-121, ResNet-50,
MobileNetV2, and EfficientNets. In this example, we will implement the learnable image
resizing module as proposed in the paper and demonstrate that on the
[Cats and Dogs dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
using the [DenseNet-121](https://arxiv.org/abs/1608.06993) architecture.

This example requires TensorFlow 2.4 or higher.
"""

"""
## Setup
"""

import npu_device
import time
import ast
import argparse

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

import numpy as np
#===============================NPU Migration=========================================
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="/home/data", type=str,
                    help='the path to train data')
parser.add_argument('--epoch', default=2, type=int,
                    help='the epoch tos train ')
parser.add_argument('--static', dest="static", type=ast.literal_eval,
                    help='Whether it is a static shape')
parser.add_argument("--log_steps", default=145, type=int,
                        help="TimeHis log Step.")
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
parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval,help='auto_tune flag, default is False')
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
  if args.auto_tune:
    npu_device.global_options().auto_tune_mode="RL,GA"
  npu_device.open().as_default()
#===============================NPU Migration=========================================

npu_config()

"""
## Define hyperparameters
"""

"""
In order to facilitate mini-batch learning, we need to have a fixed shape for the images
inside a given batch. This is why an initial resizing is required. We first resize all
the images to (300 x 300) shape and then learn their optimal representation for the
(150 x 150) resolution.
"""

INP_SIZE = (300, 300)
TARGET_SIZE = (150, 150)
INTERPOLATION = "bilinear"

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 64
EPOCHS = args.epoch

"""
In this example, we will use the bilinear interpolation but the learnable image resizer
module is not dependent on any specific interpolation method. We can also use others,
such as bicubic.
"""

"""
## Load and prepare the dataset

For this example, we will only use 40% of the total training dataset.
"""

train_ds, validation_ds = tfds.load(
    "cats_vs_dogs",
    download=False,
    data_dir=args.data_path,
    split=["train[:40%]", "train[40%:50%]"],
    as_supervised=True,
)

def preprocess_dataset(image, label):
    image = tf.image.resize(image, (INP_SIZE[0], INP_SIZE[1]))
    label = tf.one_hot(label, depth=2)
    return (image, label)


train_ds = (
    train_ds.shuffle(BATCH_SIZE * 100)
    .map(preprocess_dataset, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=args.static)
    .prefetch(AUTO)
)
validation_ds = (
    validation_ds.map(preprocess_dataset, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=args.static)
    .prefetch(AUTO)
)

"""
## Define the learnable resizer utilities

The figure below (courtesy: [Learning to Resize Images for Computer Vision Tasks](https://arxiv.org/abs/2103.09950v1))
presents the structure of the learnable resizing module:

![](https://i.ibb.co/gJYtSs0/image.png)
"""


def conv_block(x, filters, kernel_size, strides, activation=layers.LeakyReLU(0.2)):
    x = layers.Conv2D(filters, kernel_size, strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if activation:
        x = activation(x)
    return x


def res_block(x):
    inputs = x
    x = conv_block(x, 16, 3, 1)
    x = conv_block(x, 16, 3, 1, activation=None)
    return layers.Add()([inputs, x])


def get_learnable_resizer(filters=16, num_res_blocks=1, interpolation=INTERPOLATION):
    inputs = layers.Input(shape=[None, None, 3])

    # First, perform naive resizing.
    naive_resize = layers.Resizing(*TARGET_SIZE, interpolation=interpolation)(inputs)

    # First convolution block without batch normalization.
    x = layers.Conv2D(filters=filters, kernel_size=7, strides=1, padding="same")(inputs)
    x = layers.LeakyReLU(0.2)(x)

    # Second convolution block with batch normalization.
    x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)

    # Intermediate resizing as a bottleneck.
    bottleneck = layers.Resizing(*TARGET_SIZE, interpolation=interpolation)(x)

    # Residual passes.
    for _ in range(num_res_blocks):
        x = res_block(bottleneck)

    # Projection.
    x = layers.Conv2D(
        filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)

    # Skip connection.
    x = layers.Add()([bottleneck, x])

    # Final resized image.
    x = layers.Conv2D(filters=3, kernel_size=7, strides=1, padding="same")(x)
    final_resize = layers.Add()([naive_resize, x])

    return tf.keras.Model(inputs, final_resize, name="learnable_resizer")


learnable_resizer = get_learnable_resizer()


"""
## Model building utility
"""


def get_model():
    backbone = tf.keras.applications.DenseNet121(
        weights=None,
        include_top=True,
        classes=2,
        input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
    )
    backbone.trainable = True

    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
    outputs = backbone(x)

    return tf.keras.Model(inputs, outputs)


"""
The structure of the learnable image resizer module allows for flexible integrations with
different vision models.
"""

"""
## Compile and train our model with learnable resizer
"""

model = get_model()
model.compile(
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    optimizer="sgd",
    metrics=["accuracy"],
)

class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, log_steps, initial_step=0):
        self.batch_size = batch_size
        super(TimeHistory, self).__init__()
        self.steps_before_epoch = initial_step
        self.last_log_step = initial_step
        self.log_steps = log_steps
        self.steps_in_epoch = 0
        #self.opt = optimizer
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

model.fit(train_ds, validation_data=validation_ds, epochs=EPOCHS, verbose=2, callbacks=[TimeHistory(BATCH_SIZE,args.log_steps)])

"""
The plot shows that the visuals of the images have improved with training. The following
table shows the benefits of using the resizing module in comparison to using the bilinear
interpolation:

|           Model           	| Number of  parameters (Million) 	| Top-1 accuracy 	|
|:-------------------------:	|:-------------------------------:	|:--------------:	|
|   With the learnable resizer  	|             7.051717            	|      67.67%     	|
| Without the learnable resizer 	|             7.039554            	|      60.19%      	|

For more details, you can check out [this repository](https://github.com/sayakpaul/Learnable-Image-Resizing).
Note the above-reported models were trained for 10 epochs on 90% of the training set of
Cats and Dogs unlike this example. Also, note that the increase in the number of
parameters due to the resizing module is very negligible. To ensure that the improvement
in the performance is not due to stochasticity, the models were trained using the same
initial random weights.

Now, a question worth asking here is -  _isn't the improved accuracy simply a consequence
of adding more layers (the resizer is a mini network after all) to the model, compared to
the baseline?_

To show that it is not the case, the authors conduct the following experiment:

* Take a pre-trained model trained some size, say (224 x 224).

* Now, first, use it to infer predictions on images resized to a lower resolution. Record
the performance.

* For the second experiment, plug in the resizer module at the top of the pre-trained
model and warm-start the training. Record the performance.

Now, the authors argue that using the second option is better because it helps the model
learn how to adjust the representations better with respect to the given resolution.
Since the results purely are empirical, a few more experiments such as analyzing the
cross-channel interaction would have been even better. It is worth noting that elements
like [Squeeze and Excitation (SE) blocks](https://arxiv.org/abs/1709.01507), [Global Context (GC) blocks](https://arxiv.org/pdf/1904.11492) also add a few
parameters to an existing network but they are known to help a network process
information in systematic ways to improve the overall performance.
"""

"""
## Notes

* To impose shape bias inside the vision models, Geirhos et al. trained them with a
combination of natural and stylized images. It might be interesting to investigate if
this learnable resizing module could achieve something similar as the outputs seem to
discard the texture information.

* The resizer module can handle arbitrary resolutions and aspect ratios which is very
important for tasks like object detection and segmentation.

* There is another closely related topic on ***adaptive image resizing*** that attempts
to resize images/feature maps adaptively during training. [EfficientV2](https://arxiv.org/pdf/2104.00298)
uses this idea.
"""
