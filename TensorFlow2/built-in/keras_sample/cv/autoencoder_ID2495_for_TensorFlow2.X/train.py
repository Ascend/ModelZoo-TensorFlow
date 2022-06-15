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
Title: Convolutional autoencoder for image denoising
Author: [Santiago L. Valdarrama](https://twitter.com/svpino)
Date created: 2021/03/01
Last modified: 2021/03/01
Description: How to train a deep convolutional autoencoder for image denoising.
"""

"""
## Introduction

This example demonstrates how to implement a deep convolutional autoencoder
for image denoising, mapping noisy digits images from the MNIST dataset to
clean digits images. This implementation is based on an original blog post
titled [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
by [François Chollet](https://twitter.com/fchollet).
"""

"""
## Setup
"""
import npu_device
print('npu_device loaded')

import os
import ast
import time
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', default='/user/MNIST', help="""directory to data""")
    parser.add_argument('--lr', default=0.0001, type=float, help="""learning rate""")
    parser.add_argument('--batch_size', default=128, type=int, help="""batch size for 1p""")
    parser.add_argument('--epochs', default=10, type=int, help="""epochs""")
    parser.add_argument('--log_steps', default=1, type=int, help="""log frequency""")
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
    parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval, help='autotune flag, default is False')
    parser.add_argument('--static', default=0, type=int,help="""static shape""")
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

npu_config()

data_path = os.path.join(args.data_dir, 'mnist.npz')
batch_size = args.batch_size
initial_learning_rate = args.lr
epochs = args.epochs

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 28, 28, 1))
    return array

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
            #print('step:%d ,loss: %f ,time:%f'%(batch, loss, dura), flush=True)
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

# def noise(array):
#     """
#     Adds random noise to each image in the supplied array.
#     """

#     noise_factor = 0.4
#     noisy_array = array + noise_factor * np.random.normal(
#         loc=0.0, scale=1.0, size=array.shape
#     )

#     return np.clip(noisy_array, 0.0, 1.0)


# def display(array1, array2):
#     """
#     Displays ten random images from each one of the supplied arrays.
#     """

#     n = 10

#     indices = np.random.randint(len(array1), size=n)
#     images1 = array1[indices, :]
#     images2 = array2[indices, :]

#     plt.figure(figsize=(20, 4))
#     for i, (image1, image2) in enumerate(zip(images1, images2)):
#         ax = plt.subplot(2, n, i + 1)
#         plt.imshow(image1.reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)

#         ax = plt.subplot(2, n, i + 1 + n)
#         plt.imshow(image2.reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)

#     plt.show()


"""
## Prepare the data
"""

# Since we only need images from the dataset to encode and decode, we
# won't use the labels.
#(x_train, _), (x_test, _) = mnist.load_data(data_path)
(x_train, y_train), (x_test, y_test) = mnist.load_data(data_path)
if args.static==1:
        x_train, y_train = np.array(x_train[:59904], dtype='object'), y_train[:59904]
        x_test=np.array(x_test[:9984], dtype='object')
 
# Normalize and reshape the data
train_data = preprocess(x_train)
test_data = preprocess(x_test)

# Create a copy of the data with added noise
# noisy_train_data = noise(train_data)
# noisy_test_data = noise(test_data)

# Display the train data and a version of it with added noise
# display(train_data, noisy_train_data)

"""
## Build the autoencoder

We are going to use the Functional API to build our convolutional autoencoder.
"""

input = layers.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Decoder
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Autoencoder
autoencoder = Model(input, x)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
# autoencoder.summary()

"""
Now we can train our autoencoder using `train_data` as both our input data
and target. Notice we are setting up the validation data using the same
format.
"""

autoencoder.fit(
    x=train_data,
    y=train_data,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(test_data, test_data),
    callbacks=LossHistory(batch_size),
    verbose=2
)

tf.saved_model.save(autoencoder, "model_saved_model")
"""
Let's predict on our test dataset and display the original image together with
the prediction from our autoencoder.

Notice how the predictions are pretty close to the original images, although
not quite the same.
"""

# predictions = autoencoder.predict(test_data)
# display(test_data, predictions)

"""
Now that we know that our autoencoder works, let's retrain it using the noisy
data as our input and the clean data as our target. We want our autoencoder to
learn how to denoise the images.
"""

# autoencoder.fit(
#     x=noisy_train_data,
#     y=train_data,
#     epochs=100,
#     batch_size=128,
#     shuffle=True,
#     validation_data=(noisy_test_data, test_data),
# )

"""
Let's now predict on the noisy data and display the results of our autoencoder.

Notice how the autoencoder does an amazing job at removing the noise from the
input images.
"""

# predictions = autoencoder.predict(noisy_test_data)
# display(noisy_test_data, predictions)