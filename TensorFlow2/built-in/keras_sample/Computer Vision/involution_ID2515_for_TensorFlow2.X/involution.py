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

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import npu_device
import time
import tensorflow as tf
import numpy as np
from tensorflow import keras
import argparse
import ast
# import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch

# Set seed for reproducibility.
tf.random.set_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default="../cifar-10-batches-py/",
                        help="""directory to data""")
    parser.add_argument('--batch_size', default=128, type=int,
                        help="""batch size for 1p""")
    parser.add_argument('--epochs', default=10, type=int,
                        help="""epochs""")
    parser.add_argument('--Drop_Reminder', dest="Drop_Reminder", type=ast.literal_eval,
                        help='static or not')
    parser.add_argument('--save_h5', dest="save_h5", type=ast.literal_eval,
                        help='whether save h5 file after training')
    parser.add_argument('--network', default="convolution",
                        help='train network, only "convolution" or "involution"')
     #===============================NPU Migration=========================================
    parser.add_argument("--log_steps", default=50, type=int,
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

class Involution(keras.layers.Layer):
    def __init__(
            self, channel, group_number, kernel_size, stride, reduction_ratio, name
    ):
        super().__init__(name=name)

        # Initialize the parameters.
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        # Get the shape of the input.
        (_, height, width, num_channels) = input_shape

        # Scale the height and width with respect to the strides.
        height = height // self.stride
        width = width // self.stride

        # Define a layer that average pools the input tensor
        # if stride is more than 1.
        self.stride_layer = (
            keras.layers.AveragePooling2D(
                pool_size=self.stride, strides=self.stride, padding="same"
            )
            if self.stride > 1
            else tf.identity
        )
        # Define the kernel generation layer.
        self.kernel_gen = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=self.channel // self.reduction_ratio, kernel_size=1
                ),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Conv2D(
                    filters=self.kernel_size * self.kernel_size * self.group_number,
                    kernel_size=1,
                ),
            ]
        )
        # Define reshape layers
        self.kernel_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                1,
                self.group_number,
            )
        )
        self.input_patches_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                num_channels // self.group_number,
                self.group_number,
            )
        )
        self.output_reshape = keras.layers.Reshape(
            target_shape=(height, width, num_channels)
        )

    def call(self, x):
        # Generate the kernel with respect to the input tensor.
        # B, H, W, K*K*G
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)

        # reshape the kerenl
        # B, H, W, K*K, 1, G
        kernel = self.kernel_reshape(kernel)

        # Extract input patches.
        # B, H, W, K*K*C
        input_patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        # Reshape the input patches to align with later operations.
        # B, H, W, K*K, C//G, G
        input_patches = self.input_patches_reshape(input_patches)

        # Compute the multiply-add operation of kernels and patches.
        # B, H, W, K*K, C//G, G
        output = tf.multiply(kernel, input_patches)
        # B, H, W, C//G, G
        output = tf.reduce_sum(output, axis=3)

        # Reshape the output kernel.
        # B, H, W, C
        output = self.output_reshape(output)

        # Return the output tensor and the kernel.
        return output, kernel

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
 
"""
## Testing the Involution layer
"""
# Define the input tensor.
input_tensor = tf.random.normal((32, 256, 256, 3))

# Compute involution with stride 1.
output_tensor, _ = Involution(
    channel=3, group_number=1, kernel_size=5, stride=1, reduction_ratio=1, name="inv_1"
)(input_tensor)
print(f"with stride 1 ouput shape: {output_tensor.shape}")

# Compute involution with stride 2.
output_tensor, _ = Involution(
    channel=3, group_number=1, kernel_size=5, stride=2, reduction_ratio=1, name="inv_2"
)(input_tensor)
print(f"with stride 2 ouput shape: {output_tensor.shape}")

# Compute involution with stride 1, channel 16 and reduction ratio 2.
output_tensor, _ = Involution(
    channel=16, group_number=1, kernel_size=5, stride=1, reduction_ratio=2, name="inv_3"
)(input_tensor)
print(
    "with channel 16 and reduction ratio 2 ouput shape: {}".format(output_tensor.shape)
)

# Load the CIFAR10 dataset.
# TODO: download CIFAR10 dataset from blue zone and load it form local env.
def load_data(data_path):
    num_train_samples = 50000
    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(data_path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(data_path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)
    
    return (x_train, y_train), (x_test, y_test)

print("loading the CIFAR10 dataset... from local path: ", args.data_path)
(
    (train_images, train_labels),
    (test_images, test_labels,),
) = load_data(args.data_path)  # load_data("/home/hzh/involution/cifar-10-batches-py")

# Normalize pixel values to be between 0 and 1.
(train_images, test_images) = (train_images / 255.0, test_images / 255.0)

# Shuffle and batch the dataset.
train_ds = (
    tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(args.batch_size)
        .batch(args.batch_size, drop_remainder=args.Drop_Reminder)
)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(args.batch_size, drop_remainder=args.Drop_Reminder)
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

"""
## Visualise the data
"""

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

from my_adam import My_Adam
from my_losses import SparseCategoricalCrossentropy

# convolution model
if args.network == "convolution":
    # Build the conv model.
    print("building the convolution model...")
    conv_model = keras.Sequential(
        [
            keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding="same"),
            keras.layers.ReLU(name="relu1"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), padding="same"),
            keras.layers.ReLU(name="relu2"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), padding="same"),
            keras.layers.ReLU(name="relu3"),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10),
        ]
    )


    # Compile the mode with the necessary loss function and optimizer.
    print("compiling the convolution model...")
    conv_model.compile(
        #optimizer="adam",
        optimizer=My_Adam(learning_rate=0.001),
        #loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Train the model.
    print("conv model training...")
    conv_hist = conv_model.fit(train_ds, epochs=args.epochs, validation_data=test_ds, batch_size=args.batch_size, callbacks=TimeHistory(args.batch_size,args.log_steps), verbose=2)
    if args.save_h5:
        conv_model.save("model.h5")

# involution model
if args.network == "involution":
    # Build the involution model.
    # print("building the involution model...")

    inputs = keras.Input(shape=(32, 32, 3))
    x, _ = Involution(channel=3, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name="inv_1")(inputs)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x, _ = Involution(channel=3, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name="inv_2")(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x, _ = Involution(channel=3, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name="inv_3")(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)

    inv_model = keras.Model(inputs=[inputs], outputs=[outputs], name="inv_model")

    # Compile the mode with the necessary loss function and optimizer.
    print("compiling the involution model...")
    inv_model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    
    # train the model
    print("inv model training...")
    inv_hist = inv_model.fit(train_ds, epochs=args.epochs, validation_data=test_ds, batch_size=args.batch_size, callbacks=TimeHistory(args.batch_size,args.log_steps), verbose=2)
    if args.save_h5:
        inv_hist.save("inv_model.h5")
