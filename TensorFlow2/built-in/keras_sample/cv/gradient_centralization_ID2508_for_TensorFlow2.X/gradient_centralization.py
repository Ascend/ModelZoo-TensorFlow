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

Title: Gradient Centralization for Better Training Performance
Author: [Rishit Dagli](https://github.com/Rishit-dagli)
Date created: 06/18/21
Last modified: 06/18/21
Description: Implement Gradient Centralization to improve training performance of DNNs.
"""
"""
## Introduction

This example implements [Gradient Centralization](https://arxiv.org/abs/2004.01461), a
new optimization technique for Deep Neural Networks by Yong et al., and demonstrates it
on Laurence Moroney's [Horses or Humans
Dataset](https://www.tensorflow.org/datasets/catalog/horses_or_humans). Gradient
Centralization can both speedup training process and improve the final generalization
performance of DNNs. It operates directly on gradients by centralizing the gradient
vectors to have zero mean. Gradient Centralization morever improves the Lipschitzness of
the loss function and its gradient so that the training process becomes more efficient
and stable.

This example requires TensorFlow 2.2 or higher as well as `tensorflow_datasets` which can
be installed with this command:

```
pip install tensorflow-datasets
```

We will be implementing Gradient Centralization in this example but you could also use
this very easily with a package I built,
[gradient-centralization-tf](https://github.com/Rishit-dagli/Gradient-Centralization-TensorFlow).
"""

"""
## Setup
"""

#from time import time
import ast
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
import argparse
import npu_convert_dropout
from tensorflow.python.eager import context
import npu_device
import time
#npu_device.open().as_default()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default='/root/tensorflow_datasets', help="""directory to data""")
    parser.add_argument('--batch_size', default=128, type=int, help="""batch size for 1p""")
    parser.add_argument('--epochs', default=10, type=int, help="""epochs""")
    parser.add_argument('--drop_remainder', default=True, type=bool, help="""drop_remainder""")
    parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='the path to save over dump data')
    parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval,help='if or not over detection, default is False')
    parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval,help='data dump flag, default is False')
    parser.add_argument('--data_dump_step', default="10",help='data dump step, default is 10')
    parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
    parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
    parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
    parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
    parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval,help='use_mixlist flag, default is False')
    parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval,help='fusion_off flag, default is False')
    parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
    parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
    parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval,help='auto_tune flag, default is False')
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

class TimeHistory1(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, log_steps, initial_step=0):
        self.batch_size = batch_size
        super(TimeHistory1, self).__init__()
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
## Prepare the data

For this example, we will be using the [Horses or Humans
dataset](https://www.tensorflow.org/datasets/catalog/horses_or_humans).
"""

num_classes = 2
input_shape = (300, 300, 3)
dataset_name = "horses_or_humans"
batch_size = args.batch_size  # 128
AUTOTUNE = tf.data.AUTOTUNE

(train_ds, test_ds), metadata = tfds.load(
    name=dataset_name,
    split=[tfds.Split.TRAIN, tfds.Split.TEST],
    with_info=True,
    as_supervised=True,
    data_dir=args.data_path,
    download=False
)

print(f"Image shape: {metadata.features['image'].shape}")
print(f"Training images: {metadata.splits['train'].num_examples}")
print(f"Test images: {metadata.splits['test'].num_examples}")

"""
## Use Data Augmentation

We will rescale the data to `[0, 1]` and perform simple augmentations to our data.
"""

rescale = layers.Rescaling(1.0 / 255)

with context.device('CPU:0'):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.3),
            layers.RandomZoom(0.2),
        ]
    )


def prepare(ds, shuffle=False, augment=False):
    # Rescale dataset
    ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1024)

    # Batch dataset
    ds = ds.batch(batch_size, drop_remainder=args.drop_remainder)

    # Use data augmentation only on the training set
    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )

    # Use buffered prefecting
    return ds.prefetch(buffer_size=AUTOTUNE)


"""
Rescale and augment the data
"""

train_ds = prepare(train_ds, shuffle=True, augment=True)
test_ds = prepare(test_ds)

"""
## Define a model

In this section we will define a Convolutional neural network.
"""

model = tf.keras.Sequential(
    [
        layers.Conv2D(16, (3, 3), activation="relu", input_shape=(300, 300, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.Dropout(0.5),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Dropout(0.5),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

"""
## Implement Gradient Centralization

We will now
subclass the `RMSProp` optimizer class modifying the
`tf.keras.optimizers.Optimizer.get_gradients()` method where we now implement Gradient
Centralization. On a high level the idea is that let us say we obtain our gradients
through back propogation for a Dense or Convolution layer we then compute the mean of the
column vectors of the weight matrix, and then remove the mean from each column vector.

The experiments in [this paper](https://arxiv.org/abs/2004.01461) on various
applications, including general image classification, fine-grained image classification,
detection and segmentation and Person ReID demonstrate that GC can consistently improve
the performance of DNN learning.

Also, for simplicity at the moment we are not implementing gradient cliiping functionality,
however this quite easy to implement.

At the moment we are just creating a subclass for the `RMSProp` optimizer
however you could easily reproduce this for any other optimizer or on a custom
optimizer in the same way. We will be using this class in the later section when
we train a model with Gradient Centralization.
"""


class GCRMSprop(RMSprop):
    def get_gradients(self, loss, params):
        # We here just provide a modified get_gradients() function since we are
        # trying to just compute the centralized gradients.

        grads = []
        gradients = super().get_gradients()
        for grad in gradients:
            grad_len = len(grad.shape)
            if grad_len > 1:
                axis = list(range(grad_len - 1))
                grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)
            grads.append(grad)

        return grads


optimizer = GCRMSprop(learning_rate=1e-4)

"""
## Training utilities

We will also create a callback which allows us to easily measure the total training time
and the time taken for each epoch since we are interested in comparing the effect of
Gradient Centralization on the model we built above.
"""

'''
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time() - self.epoch_time_start)
'''
class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
"""
## Train the model without GC

We now train the model we built earlier without Gradient Centralization which we can
compare to the training performance of the model trained with Gradient Centralization.
"""

time_callback_no_gc = TimeHistory() 
model.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(learning_rate=1e-4),
    metrics=["accuracy"],
)

model.summary()

"""
We also save the history since we later want to compare our model trained with and not
trained with Gradient Centralization
"""
#history_no_gc = model.fit(train_ds, epochs=args.epochs, verbose=2, callbacks=[TimeHistory1(args.batch_size,8)])
history_no_gc = model.fit(train_ds, epochs=args.epochs, verbose=2, callbacks=[time_callback_no_gc,TimeHistory1(args.batch_size,8)])

"""
## Train the model with GC

We will now train the same model, this time using Gradient Centralization,
notice our optimizer is the one using Gradient Centralization this time.
"""

time_callback_gc = TimeHistory()
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

model.summary()
#history_gc = model.fit(train_ds, epochs=args.epochs, verbose=2, callbacks=[TimeHistory1(args.batch_size,8)])
history_gc = model.fit(train_ds, epochs=args.epochs, verbose=2, callbacks=[time_callback_gc,TimeHistory1(args.batch_size,8)])
model.save_weights(filepath="checkpoint/tf_model", save_format="tf")

"""
## Comparing performance
"""

print("Not using Gradient Centralization")
print(f"Loss is: {history_no_gc.history['loss'][-1]}")
print(f"Accuracy: {history_no_gc.history['accuracy'][-1]}")
print(f"Training Time: {sum(time_callback_no_gc.times)}")

print("Using Gradient Centralization")
print(f"Loss is: {history_gc.history['loss'][-1]}")
print(f"Accuracy: {history_gc.history['accuracy'][-1]}")
print(f"Training Time: {sum(time_callback_gc.times)}")

"""
Readers are encouraged to try out Gradient Centralization on different datasets from
different domains and experiment with it's effect. You are strongly advised to check out
the [original paper](https://arxiv.org/abs/2004.01461) as well - the authors present
several studies on Gradient Centralization showing how it can improve general
performance, generalization, training time as well as more efficient.

Many thanks to [Ali Mustufa Shaikh](https://github.com/ialimustufa) for reviewing this
implementation.
"""
