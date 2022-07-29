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
Title: Image classification with ConvMixer
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/10/12
Last modified: 2021/10/12
Description: An all-convolutional network applied to patches of images.
"""
"""
## Introduction

Vision Transformers (ViT; [Dosovitskiy et al.](https://arxiv.org/abs/1612.00593)) extract
small patches from the input images, linearly project them, and then apply the
Transformer ([Vaswani et al.](https://arxiv.org/abs/1706.03762)) blocks. The application
of ViTs to image recognition tasks is quickly becoming a promising area of research,
because ViTs eliminate the need to have strong inductive biases (such as convolutions) for
modeling locality. This presents them as a general computation primititive capable of
learning just from the training data with as minimal inductive priors as possible. ViTs
yield great downstream performance when trained with proper regularization, data
augmentation, and relatively large datasets.

In the [Patches Are All You Need](https://openreview.net/pdf?id=TVHS5Y4dNvM) paper (note: at
the time of writing, it is a submission to the ICLR 2022 conference), the authors extend
the idea of using patches to train an all-convolutional network and demonstrate
competitive results. Their architecture namely **ConvMixer** uses recipes from the recent
isotrophic architectures like ViT, MLP-Mixer
([Tolstikhin et al.](https://arxiv.org/abs/2105.01601)), such as using the same
depth and resolution across different layers in the network, residual connections,
and so on.

In this example, we will implement the ConvMixer model and demonstrate its performance on
the CIFAR-10 dataset.

To use the AdamW optimizer, we need to install TensorFlow Addons:

```shell
pip install -U -q tensorflow-addons
```
"""

"""
## Imports
"""
import npu_device
import time
import os
import ast
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.eager import context
# import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import argparse
"""
## Hyperparameters

To keep run time short, we will train the model for only 10 epochs. To focus on
the core ideas of ConvMixer, we will not use other training-specific elements like
RandAugment ([Cubuk et al.](https://arxiv.org/abs/1909.13719)). If you are interested in
learning more about those details, please refer to the
[original paper](https://openreview.net/pdf?id=TVHS5Y4dNvM).
"""
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', default="../cifar-10-batches-py/",
                        help="""directory to data""")
    parser.add_argument('--lr', default=0.001, type=float,
                        help="""learning rate""")
    parser.add_argument('--weight_decay', default=0.0001, type=float,
                        help="""weight decay""")
    parser.add_argument('--batch_size', default=128, type=int,
                        help="""batch size for 1p""")
    parser.add_argument('--epochs', default=10, type=int,
                        help="""epochs""")
    parser.add_argument('--eval_static', dest="eval_static", type=ast.literal_eval,
                        help='eval static or not')
    parser.add_argument('--force', dest="force", type=ast.literal_eval,
                        help='force preprocessing on CPU')
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
    parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval, help='auto_tune flag, default is False')
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args

args = parse_args()
data_path = args.data_dir
learning_rate = args.lr
weight_decay = args.weight_decay
batch_size = args.batch_size
num_epochs = args.epochs 

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

print('npu_device loaded')
npu_config()

"""
## Load the CIFAR-10 dataset
"""
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

(x_train, y_train), (x_test, y_test) = load_data(data_path)

val_split = 0.1

val_indices = int(len(x_train) * val_split)
new_x_train, new_y_train = x_train[val_indices:], y_train[val_indices:]
x_val, y_val = x_train[:val_indices], y_train[:val_indices]

print(f"Training data samples: {len(new_x_train)}")
print(f"Validation data samples: {len(x_val)}")
print(f"Test data samples: {len(x_test)}")

"""
## Prepare `tf.data.Dataset` objects

Our data augmentation pipeline is different from what the authors used for the CIFAR-10
dataset, which is fine for the purpose of the example.
"""

image_size = 32
auto = tf.data.AUTOTUNE
if args.force:
    with context.device('CPU:0'):
        data_augmentation = keras.Sequential(
            [layers.RandomCrop(image_size, image_size), layers.RandomFlip("horizontal"),],
            name="data_augmentation",
        )
else:
    data_augmentation = keras.Sequential(
        [layers.RandomCrop(image_size, image_size), layers.RandomFlip("horizontal"),],
        name="data_augmentation",
    )

def make_datasets(images, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    if args.eval_static:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        dataset = dataset.batch(batch_size)
    if is_train:
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x), y), num_parallel_calls=auto
        )
    return dataset.prefetch(auto)


train_dataset = make_datasets(new_x_train, new_y_train, is_train=True)
val_dataset = make_datasets(x_val, y_val)
test_dataset = make_datasets(x_test, y_test)

"""
## ConvMixer utilities

The following figure (taken from the original paper) depicts the ConvMixer model:

![](https://i.imgur.com/yF8actg.png)

ConvMixer is very similar to the MLP-Mixer, model with the following key
differences:

* Instead of using fully-connected layers, it uses standard convolution layers.
* Instead of LayerNorm (which is typical for ViTs and MLP-Mixers), it uses BatchNorm.

Two types of convolution layers are used in ConvMixer. **(1)**: Depthwise convolutions,
for mixing spatial locations of the images, **(2)**: Pointwise convolutions (which follow
the depthwise convolutions), for mixing channel-wise information across the patches.
Another keypoint is the use of *larger kernel sizes* to allow a larger receptive field.
"""
#@ops.RegisterGradient("FastGelu")
def _fast_gelu_grad(op,grad):
  """ The gradient for fastgelu

  Args:
    op:The fastgelu operations that we are differentiating,which we can us to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the fast_gelu op.

  Returns:
    Gradient with respect to the input of fast_gelu
  """
  return [npu_device.gen_npu_ops.fast_gelu_grad(grad,op.inputs[0])]

grad_registry_list = ops.gradient_registry.list()
if not hasattr(npu_device.ops, 'gelu') and "FastGelu" not in grad_registry_list:
  ops.RegisterGradient("FastGelu")(_fast_gelu_grad)

def activation_block(x):
    #x = layers.Activation("gelu")(x)
    if not hasattr(npu_device.ops, 'gelu'):
      x = npu_device.gen_npu_ops.fast_gelu(x)
    else:
      fast_gelu = getattr(npu_device.ops, 'gelu')
      x = fast_gelu(x)
    #x=npu_device.gen_npu_ops.fast_gelu(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depthwise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Pointwise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def get_conv_mixer_256_8(
    image_size=32, filters=256, depth=8, kernel_size=5, patch_size=2, num_classes=10
):
    """ConvMixer-256/8: https://openreview.net/pdf?id=TVHS5Y4dNvM.
    The hyperparameter values are taken from the paper.
    """
    inputs = keras.Input((image_size, image_size, 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Extract patch embeddings.
    x = conv_stem(x, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


"""
The model used in this experiment is termed as **ConvMixer-256/8** where 256 denotes the
number of channels and 8 denotes the depth. The resulting model only has 0.8 million
parameters.
"""

"""
## Model training and evaluation utility
"""

# Code reference:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/.

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

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=[checkpoint_callback,TimeHistory(batch_size,args.log_steps)],
        verbose=2
    )

    model.load_weights(checkpoint_filepath)
    # _, accuracy = model.evaluate(test_dataset)
    # print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, model


"""
## Train and evaluate model
"""

conv_mixer_model = get_conv_mixer_256_8()
history, conv_mixer_model = run_experiment(conv_mixer_model)

"""
The gap in training and validation performance can be mitigated by using additional
regularization techniques. Nevertheless, being able to get to ~83% accuracy within 10
epochs with 0.8 million parameters is a strong result.
"""

"""
## Visualizing the internals of ConvMixer

We can visualize the patch embeddings and the learned convolution filters. Recall
that each patch embedding and intermediate feature map have the same number of channels
(256 in this case). This will make our visualization utility easier to implement.
"""

# Code reference: https://bit.ly/3awIRbP.


# def visualization_plot(weights, idx=1):
#     # First, apply min-max normalization to the
#     # given weights to avoid isotrophic scaling.
#     p_min, p_max = weights.min(), weights.max()
#     weights = (weights - p_min) / (p_max - p_min)

#     # Visualize all the filters.
#     num_filters = 256
#     plt.figure(figsize=(8, 8))

#     for i in range(num_filters):
#         current_weight = weights[:, :, :, i]
#         if current_weight.shape[-1] == 1:
#             current_weight = current_weight.squeeze()
#         ax = plt.subplot(16, 16, idx)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         plt.imshow(current_weight)
#         idx += 1


# We first visualize the learned patch embeddings.
# patch_embeddings = conv_mixer_model.layers[2].get_weights()[0]
# visualization_plot(patch_embeddings)

"""
Even though we did not train the network to convergence, we can notice that different
patches show different patterns. Some share similarity with others while some are very
different. These visualizations are more salient with larger image sizes.

Similarly, we can visualize the raw convolution kernels. This can help us understand
the patterns to which a given kernel is receptive.
"""

# First, print the indices of the convolution layers that are not
# pointwise convolutions.
# for i, layer in enumerate(conv_mixer_model.layers):
#     if isinstance(layer, layers.DepthwiseConv2D):
#         if layer.get_config()["kernel_size"] == (5, 5):
#             print(i, layer)

# idx = 26  # Taking a kernel from the middle of the network.

# kernel = conv_mixer_model.layers[idx].get_weights()[0]
# kernel = np.expand_dims(kernel.squeeze(), axis=2)
# visualization_plot(kernel)

"""
We see that different filters in the kernel have different locality spans, and this pattern
is likely to evolve with more training.
"""

"""
## Final notes

There's been a recent trend on fusing convolutions with other data-agnostic operations
like self-attention. Following works are along this line of research:

* ConViT ([d'Ascoli et al.](https://arxiv.org/abs/2103.10697))
* CCT ([Hassani et al.](https://arxiv.org/abs/2104.05704))
* CoAtNet ([Dai et al.](https://arxiv.org/abs/2106.04803))
"""