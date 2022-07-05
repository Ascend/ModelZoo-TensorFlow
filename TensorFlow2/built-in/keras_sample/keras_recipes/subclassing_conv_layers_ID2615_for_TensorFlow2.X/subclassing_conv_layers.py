"""
Title: Customizing the convolution operation of a Conv2D layer
Author: [lukewood](https://lukewood.xyz)
Date created: 11/03/2021
Last modified: 11/03/2021
Description: This example shows how to implement custom convolution layers using the `Conv.convolution_op()` API.
"""
"""
## Introduction

You may sometimes need to implement custom versions of convolution layers like `Conv1D` and `Conv2D`.
Keras enables you do this without implementing the entire layer from scratch: you can reuse
most of the base convolution layer and just customize the convolution op itself via the
`convolution_op()` method.

This method was introduced in Keras 2.7. So before using the
`convolution_op()` API, ensure that you are running Keras version 2.7.0 or greater.
"""
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
import tensorflow.keras as keras

print(keras.__version__)
"""
## A Simple `StandardizedConv2D` implementation

There are two ways to use the `Conv.convolution_op()` API. The first way
is to override the `convolution_op()` method on a convolution layer subclass.
Using this approach, we can quickly implement a
[StandardizedConv2D](https://arxiv.org/abs/1903.10520) as shown below.
"""
import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers
import numpy as np
from time import time
import npu_device
import os
import time
from absl import flags, app
import npu_convert_dropout


flags.DEFINE_string(name='data_path', default='/home/hzh/involution/cifar-10-batches-py',
                    help='dataset path(local)')
flags.DEFINE_integer(name='epochs', default=5, help='training epochs')
flags.DEFINE_integer(name='batch_size', default=128, help='training batch_size')
flags.DEFINE_boolean(name='save_h5', default=True, help='whether save h5 file after training')
flags.DEFINE_integer(name='log_steps', default=234, help='training epochs')
flags.DEFINE_string(name='precision_mode', default= 'allow_fp32_to_fp16',
                    help='allow_fp32_to_fp16/force_fp16/ ' 
                    'must_keep_origin_dtype/allow_mix_precision.')
flags.DEFINE_boolean(name='over_dump', default=False,
                    help='if or not over detection, default is False')
flags.DEFINE_boolean(name='data_dump_flag', default=False,
                    help='data dump flag, default is False')
flags.DEFINE_string(name='data_dump_step', default="10",
                    help='data dump step, default is 10')
flags.DEFINE_boolean(name='profiling', default=False,
                    help='if or not profiling for performance debug, default is False') 
flags.DEFINE_string(name='profiling_dump_path', default="/home/data",
                    help='the path to save profiling data')                                      
flags.DEFINE_string(name='over_dump_path', default="/home/data",
                    help='the path to save over dump data')  
flags.DEFINE_string(name='data_dump_path', default="/home/data",
                    help='the path to save dump data') 
flags.DEFINE_boolean(name='use_mixlist', default=False,
                    help='whether to enable mixlist, default is True')
flags.DEFINE_boolean(name='fusion_off_flag', default=False,
                    help='whether to enable mixlist, default is True')
flags.DEFINE_string(name='mixlist_file', default='ops_info.json',
                    help='mixlist file name, default is ops_info.json')
flags.DEFINE_string(name='fusion_off_file', default='fusion_switch.cfg',
                    help='fusion_off file name, default is fusion_switch.cfg')
flags.DEFINE_boolean(name='auto_tune', default=False,
                    help='auto_tune flag, default is False')
flags.DEFINE_integer(name='static', default=0,
                    help='static, default is 0')
FLAGS = flags.FLAGS

def npu_config():

    
    npu_config = {}

    if FLAGS.data_dump_flag:
        npu_device.global_options().dump_config.enable_dump = True
        npu_device.global_options().dump_config.dump_path = FLAGS.data_dump_path
        npu_device.global_options().dump_config.dump_step = FLAGS.data_dump_step
        npu_device.global_options().dump_config.dump_mode = "all"

    if FLAGS.over_dump:
        npu_device.global_options().dump_config.enable_dump_debug = True
        npu_device.global_options().dump_config.dump_path = FLAGS.over_dump_path
        npu_device.global_options().dump_config.dump_debug_mode = "all"

    if FLAGS.profiling:
        npu_device.global_options().profiling_config.enable_profiling = True
        profiling_options = '{"output":"' + FLAGS.profiling_dump_path + '", \
                            "training_trace":"on", \
                            "task_trace":"on", \
                            "aicpu":"on", \
                            "aic_metrics":"PipeUtilization",\
                            "fp_point":"", \
                            "bp_point":""}'
        npu_device.global_options().profiling_config.profiling_options = profiling_options
    npu_device.global_options().precision_mode=FLAGS.precision_mode
    if FLAGS.use_mixlist and FLAGS.precision_mode=='allow_mix_precision':
        npu_device.global_options().modify_mixlist=FLAGS.mixlist_file
    if FLAGS.fusion_off_flag:
        npu_device.global_options().fusion_switch_file=FLAGS.fusion_off_file
    if FLAGS.auto_tune:
        npu_device.global_options().auto_tune_mode="RL,GA"
    npu_device.open().as_default()
#===============================NPU Migration=========================================

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


def task(_):
    class StandardizedConv2DWithOverride(layers.Conv2D):
        def convolution_op(self, inputs, kernel):
            mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
            return tf.nn.conv2d(
                inputs,
                (kernel - mean) / tf.sqrt(var + 1e-10),
                padding="VALID",
                strides=list(self.strides),
                name=self.__class__.__name__,
            )


    """
    The other way to use the `Conv.convolution_op()` API is to directly call the
    `convolution_op()` method from the `call()` method of a convolution layer subclass.
    A comparable class implemented using this approach is shown below.
    """


    class StandardizedConv2DWithCall(layers.Conv2D):
        def convolution_op(self, inputs, kernel):
            mean, var = tf.nn.moments(kernel, axes=[0, 1, 2], keepdims=True)
            return tf.nn.conv2d(
                inputs,
                (kernel - mean) / tf.sqrt(var + 1e-10),
                padding="VALID",
                strides=list(self.strides),
                name=self.__class__.__name__,
            )

        def call(self, inputs):
            mean, var = tf.nn.moments(self.kernel, axes=[0, 1, 2], keepdims=True)
            result = self.convolution_op(
                inputs, (self.kernel - mean) / tf.sqrt(var + 1e-10)
            )
            if self.use_bias:
                result = result + self.bias
            return result


    """
    ## Example Usage
    
    Both of these layers work as drop-in replacements for `Conv2D`. The following
    demonstration performs classification on the MNIST dataset.
    """

    npu_config()

    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1)
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(os.path.join(FLAGS.data_path, 'mnist.npz'))

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    if FLAGS.static==1:
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .batch(batch_size, drop_remainder=True))
    else: 
        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .batch(batch_size, drop_remainder=False))
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=input_shape),
            StandardizedConv2DWithCall(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            StandardizedConv2DWithOverride(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.summary()
    """
    
    """


    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    callbacks = [TimeHistory(batch_size,FLAGS.log_steps)]
    #start_time = time()
    #model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2)
    model.fit(train_ds, batch_size=batch_size, epochs=epochs,  verbose=2, callbacks=callbacks)
    #end_time = time()
    #time_s = end_time - start_time
    #print("TrainingTime: ", time_s)
    
    if FLAGS.save_h5:
        model.save("model.h5")
    """
    ## Conclusion
    
    The `Conv.convolution_op()` API provides an easy and readable way to implement custom
    convolution layers. A `StandardizedConvolution` implementation using the API is quite
    terse, consisting of only four lines of code.
    """


if __name__ == '__main__':
    app.run(task)
