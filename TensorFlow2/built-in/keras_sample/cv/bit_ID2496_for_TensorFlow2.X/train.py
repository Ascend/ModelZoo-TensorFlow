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
Title: Image Classification using BigTransfer (BiT)
Author: [Sayan Nath](https://twitter.com/sayannath2350)
Date created: 2021/09/24
Last modified: 2021/09/24
Description: BigTransfer (BiT) State-of-the-art transfer learning for image classification.
"""

"""
## Introduction

BigTransfer (also known as BiT) is a state-of-the-art transfer learning method for image
classification. Transfer of pre-trained representations improves sample efficiency and
simplifies hyperparameter tuning when training deep neural networks for vision. BiT
revisit the paradigm of pre-training on large supervised datasets and fine-tuning the
model on a target task. The importance of appropriately choosing normalization layers and
scaling the architecture capacity as the amount of pre-training data increases.

BigTransfer(BiT) is trained on public datasets, along with code in
[TF2, Jax and Pytorch](https://github.com/google-research/big_transfer). This will help anyone to reach
state of the art performance on their task of interest, even with just a handful of
labeled images per class.

You can find BiT models pre-trained on
[ImageNet](https://image-net.org/challenges/LSVRC/2012/index) and ImageNet-21k in
[TFHub](https://tfhub.dev/google/collections/bit/1) as TensorFlow2 SavedModels that you
can use easily as Keras Layers. There are a variety of sizes ranging from a standard
ResNet50 to a ResNet152x4 (152 layers deep, 4x wider than a typical ResNet50) for users
with larger computational and memory budgets but higher accuracy requirements.

![](https://i.imgur.com/XeWVfe7.jpeg)
Figure: The x-axis shows the number of images used per class, ranging from 1 to the full
dataset. On the plots on the left, the curve in blue above is our BiT-L model, whereas
the curve below is a ResNet-50 pre-trained on ImageNet (ILSVRC-2012).
"""

"""
## Setup
"""
import npu_device

import os
import ast
import time
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

SEEDS = 42

np.random.seed(SEEDS)
tf.random.set_seed(SEEDS)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_steps', default=1, type=int, help="""log frequency""")
    parser.add_argument('--data_dir', default="../bit_datasets/", help="""directory to data""")
    parser.add_argument('--batch_size', default=64, type=int, help="""batch size for 1p""")
    parser.add_argument('--epochs', default=30, type=int, help="""epochs""")
    parser.add_argument('--eval_static', dest="eval_static", type=ast.literal_eval, help='the path to train data')
    parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='the path to save over dump data')
    parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval, help='if or not over detection, default is False')
    parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval, help='data dump flag, default is False')
    parser.add_argument('--data_dump_step', default="10", help='data dump step, default is 10')
    parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
    parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
    parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
    parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
    parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval, help='use_mixlist flag, default is False')
    parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval, help='fusion_off flag, default is False')
    parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
    parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args

args = parse_args()

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
  npu_device.global_options().precision_mode=args.precision_mode
  if args.use_mixlist and args.precision_mode=='allow_mix_precision':
    npu_device.global_options().modify_mixlist=args.mixlist_file
  if args.fusion_off_flag:
    npu_device.global_options().fusion_switch_file=args.fusion_off_file
  npu_device.open().as_default()
npu_config()

data_path = args.data_dir
bit_model_url = os.path.join(data_path, 'ttst')
num_epochs = args.epochs
"""
## Gather Flower Dataset
"""

train_ds, validation_ds = tfds.load(
    "tf_flowers", split=["train[:85%]", "train[85%:]"], download=False,
    data_dir=data_path, as_supervised=True,
)

"""
## Visualise the dataset
"""

# plt.figure(figsize=(10, 10))
# for i, (image, label) in enumerate(train_ds.take(9)):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(image)
#     plt.title(int(label))
#     plt.axis("off")

"""
## Define hyperparameters
"""

RESIZE_TO = 384
CROP_TO = 224
BATCH_SIZE = args.batch_size
STEPS_PER_EPOCH = 10
AUTO = tf.data.AUTOTUNE  # optimise the pipeline performance
NUM_CLASSES = 5  # number of classes
SCHEDULE_LENGTH = (
    500  # we will train on lower resolution images and will still attain good results
)
SCHEDULE_BOUNDARIES = [
    200,
    300,
    400,
]  # more the dataset size the schedule length increase

"""
The hyperparamteres like `SCHEDULE_LENGTH` and `SCHEDULE_BOUNDARIES` are determined based
on empirical results. The method has been explained in the [original
paper](https://arxiv.org/abs/1912.11370) and in their [Google AI Blog
Post](https://ai.googleblog.com/2020/05/open-sourcing-bit-exploring-large-scale.html).

The `SCHEDULE_LENGTH` is aslo determined whether to use [MixUp
Augmentation](https://arxiv.org/abs/1710.09412) or not. You can also find an easy MixUp
Implementation in [Keras Coding Examples](https://keras.io/examples/vision/mixup/).

![](https://i.imgur.com/oSaIBYZ.jpeg)
"""

"""
## Define preprocessing helper functions
"""

SCHEDULE_LENGTH = SCHEDULE_LENGTH * 512 / BATCH_SIZE


@tf.function
def preprocess_train(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, (RESIZE_TO, RESIZE_TO))
    image = tf.image.random_crop(image, (CROP_TO, CROP_TO, 3))
    image = image / 255.0
    return (image, label)


@tf.function
def preprocess_test(image, label):
    image = tf.image.resize(image, (RESIZE_TO, RESIZE_TO))
    image = image / 255.0
    return (image, label)


DATASET_NUM_TRAIN_EXAMPLES = train_ds.cardinality().numpy()

repeat_count = int(
    SCHEDULE_LENGTH * BATCH_SIZE / DATASET_NUM_TRAIN_EXAMPLES * STEPS_PER_EPOCH
)
repeat_count += 50 + 1  # To ensure at least there are 50 epochs of training

"""
## Define the data pipeline
"""
if args.eval_static:
    # Training pipeline
    pipeline_train = (
        train_ds.shuffle(10000)
        .repeat(repeat_count)  # Repeat dataset_size / num_steps
        .map(preprocess_train, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTO)
    )

    # Validation pipeline
    pipeline_validation = (
        validation_ds.map(preprocess_test, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTO)
    )
else:
    # Training pipeline
    pipeline_train = (
        train_ds.shuffle(10000)
        .repeat(repeat_count)  # Repeat dataset_size / num_steps
        .map(preprocess_train, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    # Validation pipeline
    pipeline_validation = (
        validation_ds.map(preprocess_test, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self, bs):
        super().__init__()
        self.batch_size = bs
    def on_batch_begin(self, batch, logs={}):
        self.start = time.time()
    def on_batch_end(self, batch, logs={}):
        if batch % args.log_steps == 0:
            loss = logs.get('loss')
            dura = time.time() - self.start
            if dura < 10:
                self.epoch_perf.append(dura)
            print('step:%d ,loss: %f ,time:%f'%(batch, loss, dura), flush=True)
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_perf = []
        self.epochstart = time.time()
    def on_epoch_end(self, epoch, logs={}):
        duration = time.time() - self.epochstart
        print('epoch_duration: ', duration)
        if epoch != 0:
            self.perf.append(np.mean(self.epoch_perf))
    def on_train_begin(self, logs={}):
        print('params: ', self.params)
        self.perf = []
    def on_train_end(self, logs={}):
        print('imgs/s: %.2f'%(self.batch_size / np.mean(self.perf)))

logger = LossHistory(args.batch_size)
"""
## Visualise the training samples
"""

# image_batch, label_batch = next(iter(pipeline_train))

# plt.figure(figsize=(10, 10))
# for n in range(25):
#     ax = plt.subplot(5, 5, n + 1)
#     plt.imshow(image_batch[n])
#     plt.title(label_batch[n].numpy())
#     plt.axis("off")

"""
## Load pretrained TF-Hub model into a `KerasLayer`
"""

# bit_model_url = "https://tfhub.dev/google/bit/m-r50x1/1"
bit_module = hub.KerasLayer(bit_model_url)

"""
## Create BigTransfer (BiT) model

To create the new model, we:

1. Cut off the BiT model’s original head. This leaves us with the “pre-logits” output.
We do not have to do this if we use the ‘feature extractor’ models (i.e. all those in
subdirectories titled `feature_vectors`), since for those models the head has already
been cut off.

2. Add a new head with the number of outputs equal to the number of classes of our new
task. Note that it is important that we initialise the head to all zeroes.
"""


class MyBiTModel(keras.Model):
    def __init__(self, num_classes, module, **kwargs):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.head = keras.layers.Dense(num_classes, kernel_initializer="zeros")
        self.bit_model = module

    def call(self, images):
        bit_embedding = self.bit_model(images)
        return self.head(bit_embedding)


model = MyBiTModel(num_classes=NUM_CLASSES, module=bit_module)

"""
## Define optimizer and loss
"""

learning_rate = 0.003 * BATCH_SIZE / 512

# Decay learning rate by a factor of 10 at SCHEDULE_BOUNDARIES.
lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=SCHEDULE_BOUNDARIES,
    values=[
        learning_rate,
        learning_rate * 0.1,
        learning_rate * 0.01,
        learning_rate * 0.001,
    ],
)
optimizer = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

"""
## Compile the model
"""

model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

"""
## Set up callbacks
"""

train_callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=2, restore_best_weights=True
    ),
    logger
]

"""
## Train the model
"""

history = model.fit(
    pipeline_train,
    batch_size=BATCH_SIZE,
    epochs=num_epochs,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=pipeline_validation,
    callbacks=train_callbacks,
    verbose=2
)

"""
## Plot the training and validation metrics
"""


# def plot_hist(hist):
#     plt.plot(hist.history["accuracy"])
#     plt.plot(hist.history["val_accuracy"])
#     plt.plot(hist.history["loss"])
#     plt.plot(hist.history["val_loss"])
#     plt.title("Training Progress")
#     plt.ylabel("Accuracy/Loss")
#     plt.xlabel("Epochs")
#     plt.legend(["train_acc", "val_acc", "train_loss", "val_loss"], loc="upper left")
#     plt.show()


# plot_hist(history)

"""
## Evaluate the model
"""

# accuracy = model.evaluate(pipeline_validation)[1] * 100
# print("Accuracy: {:.2f}%".format(accuracy))

"""
## Conclusion

BiT performs well across a surprisingly wide range of data regimes
-- from 1 example per class to 1M total examples. BiT achieves 87.5% top-1 accuracy on
ILSVRC-2012, 99.4% on CIFAR-10, and 76.3% on the 19 task Visual Task Adaptation Benchmark
(VTAB). On small datasets, BiT attains 76.8% on ILSVRC-2012 with 10 examples per class,
and 97.0% on CIFAR-10 with 10 examples per class.

![](https://i.imgur.com/b1Lw5fz.png)

You can experiment further with the BigTransfer Method by following the
[original paper](https://arxiv.org/abs/1912.11370).
"""