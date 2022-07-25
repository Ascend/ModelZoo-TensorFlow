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
Title: Image classification with Swin Transformers
Author: [Rishit Dagli](https://twitter.com/rishit_dagli)
Date created: 2021/09/08
Last modified: 2021/09/08
Description: Image classification using Swin Transformers, a general-purpose backbone for computer vision.
"""
"""
This example implements [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
by Liu et al. for image classification, and demonstrates it on the
[CIFAR-100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

Swin Transformer (**S**hifted **Win**dow Transformer) can serve as a general-purpose backbone
for computer vision. Swin Transformer is a hierarchical Transformer whose
representations are computed with _shifted windows_. The shifted window scheme
brings greater efficiency by limiting self-attention computation to
non-overlapping local windows while also allowing for cross-window connections.
This architecture has the flexibility to model information at various scales and has
a linear computational complexity with respect to image size.

This example requires TensorFlow 2.5 or higher, as well as TensorFlow Addons,
which can be installed using the following commands:
"""

"""shell
pip install -U tensorflow-addons
"""

"""
## Setup
"""
import ast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=40, type=int, help="train epochs")
parser.add_argument("--batch_size", default=128, type=int, help="global batch_size")
parser.add_argument("--rank_size", default=1, type=int, help="rank size")
parser.add_argument("--device_id", default=0, type=int, help="Ascend device id")
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

import npu_device

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

npu_config()

#import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras import layers
from cifar100 import load_data

from tensorflow.python.ops.numpy_ops import np_random
import npu_convert_dropout
from tensorflow.python.framework import ops
from npu_device.npu_device import gen_npu_ops as npu_aicore_ops
"""
## Prepare the data

We load the CIFAR-100 dataset through `tf.keras.datasets`,
normalize the images, and convert the integer labels to one-hot encoded vectors.
"""



num_classes = 100
input_shape = (32, 32, 3)
batch_size = args.batch_size
rank_size = args.rank_size
device_id = args.device_id

#(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
(x_train, y_train), (x_test, y_test) = load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
##################
x_trains = np.split(x_train, rank_size)
y_trains = np.split(y_train, rank_size)
x_train = x_trains[device_id]
y_train = y_trains[device_id]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train
print(f'x_train.shape:{x_train.shape}')
batch_size = batch_size // rank_size
print(f'batch_size:{batch_size}')

##################

######################################################
#train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#train_data.shuffle(x_train.shape[0])
#train_data, batch_size = npu_device.distribute.shard_and_rebatch_dataset(train_data,batch_size)  #128
#print(f'batch_size:{batch_size}')  # 16
#train_data = train_data.batch(batch_size, drop_remainder=True)
#train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#def processing(x, y):
#    return (x, y)
#train_data = train_data.map(processing)
######################################################

#plt.figure(figsize=(10, 10))
#for i in range(25):
#    plt.subplot(5, 5, i + 1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(x_train[i])
#plt.show()

"""
## Configure the hyperparameters

A key parameter to pick is the `patch_size`, the size of the input patches.
In order to use each pixel as an individual input, you can set `patch_size` to `(1, 1)`.
Below, we take inspiration from the original paper settings
for training on ImageNet-1K, keeping most of the original settings for this example.
"""

patch_size = (2, 2)  # 2-by-2 sized patches
dropout_rate = 0.03  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64  # Embedding dimension
num_mlp = 256  # MLP layer size
qkv_bias = True  # Convert embedded patches to query, key, and values with a learnable additive value
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
image_dimension = 32  # Initial image size

#num_patch_x = input_shape[0] // patch_size[0]
#num_patch_y = input_shape[1] // patch_size[1]
num_patch_x = image_dimension // patch_size[0]
num_patch_y = image_dimension // patch_size[1]

learning_rate = 1e-3

#num_epochs = 40
num_epochs = args.epochs
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.1

"""
## Helper functions

We create two helper functions to help us get a sequence of
patches from the image, merge patches, and apply dropout.
"""

def my_roll(x,shift_size):
    a = x[ : , : shift_size, : , : ]
    b = x[ : , shift_size : , : , :]
    shift_1 = tf.concat([b,a], axis=1)
    c = shift_1[:, :, : shift_size, :]
    d = shift_1[:, :, shift_size :, :]
    shifted_x = tf.concat([d, c], axis=2)
    return shifted_x

def my_random_crop(image, size):
    w, h = size[0], size[1]
    W, H = image.shape[1],image.shape[2]
    delta_w = W - w + 1
    print(delta_w)
    delta_h = H - h + 1
#    ws = tf.cond(delta_w > 0, lambda: np_random.randint(delta_w), lambda:delta_w)
#    wh = tf.cond(delta_h > 0, lambda: np_random.randint(delta_h), lambda:delta_h)
    ws = np_random.randint(delta_w)
    hs = np_random.randint(delta_h)
    crop = image[:,ws:ws+w,hs:hs +h,:]
    print(ws, hs)
    return crop

def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output


"""
## Window based multi-head self-attention

Usually Transformers perform global self-attention, where the relationships between
a token and all other tokens are computed. The global computation leads to quadratic
complexity with respect to the number of tokens. Here, as the [original paper](https://arxiv.org/abs/2103.14030)
suggests, we compute self-attention within local windows, in a non-overlapping manner.
Global self-attention leads to quadratic computational complexity in the number of patches,
whereas window-based self-attention leads to linear complexity and is easily scalable.
"""


class WindowAttention(layers.Layer):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
    ):
        super(WindowAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )
        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.relative_position_index = tf.Variable(
            initial_value=tf.convert_to_tensor(relative_position_index), trainable=False
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))
            attn = keras.activations.softmax(attn, axis=-1)
        else:
            attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv


"""
## The complete Swin Transformer model

Finally, we put together the complete Swin Transformer by replacing the standard multi-head
attention (MHA) with shifted windows attention. As suggested in the
original paper, we create a model comprising of a shifted window-based MHA
layer, followed by a 2-layer MLP with GELU nonlinearity in between, applying
`LayerNormalization` before each MSA layer and each MLP, and a residual
connection after each of these layers.

Notice that we only create a simple MLP with 2 Dense and
2 Dropout layers. Often you will see models using ResNet-50 as the MLP which is
quite standard in the literature. However in this paper the authors use a
2-layer MLP with GELU nonlinearity in between.
"""
#@ops.RegisterGradient("FastGelu")
def _fast_gelu_grad(op,grad):
    return [npu_aicore_ops.fast_gelu_grad(grad,op.inputs[0])]

grad_registry_list = ops.gradient_registry.list()
if not hasattr(npu_device.ops, 'gelu') and "FastGelu" not in grad_registry_list:
  ops.RegisterGradient("FastGelu")(_fast_gelu_grad)

@tf.keras.utils.register_keras_serializable(package='Text')
def gelu(x):
    return npu_aicore_ops.fast_gelu(x)

class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(SwinTransformer, self).__init__(**kwargs)

        self.dim = dim  # number of input dimensions
        self.num_patch = num_patch  # number of embedded patches
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # size of window
        self.shift_size = shift_size  # size of window shift
        self.num_mlp = num_mlp  # number of MLP nodes

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(gelu),
                #layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_array = tf.convert_to_tensor(mask_array)

            # mask array to windows
            mask_windows = window_partition(mask_array, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(initial_value=attn_mask, trainable=False)

    def call(self, x):
        height, width = self.num_patch
        _, num_patches_before, channels = x.shape
        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))
        if self.shift_size > 0:
            #shifted_x = tf.roll(
            #    x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            #)
            shifted_x = my_roll(x, self.shift_size)
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )
        if self.shift_size > 0:
            #x = tf.roll(
            #    shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            #)
            x = my_roll(shifted_x, -self.shift_size) 
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x


"""
## Model training and evaluation

### Extract and embed patches

We first create 3 layers to help us extract, embed and merge patches from the
images on top of which we will later use the Swin Transformer class we built.
"""


class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[0]

    def call(self, images):
        print(f"PatchExtract-image-shape:{images.get_shape()}")
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        print(f'patches.shape:{patches.shape}')
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        patch_num_2 = patches.shape[2]
        return tf.reshape(patches, (batch_size, patch_num * patch_num_2, patch_dim))


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super(PatchMerging, self).__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height//2, 2, width//2, 2, C))
        #x0 = tf.strided_slice(x, [0,0,0,0],[128,16,16,64],[1,2,2,1])
        #x1 = tf.strided_slice(x, [0,1,0,0],[128,16,16,64],[1,2,2,1])
        #x2 = tf.strided_slice(x, [0,0,1,0],[128,16,16,64],[1,2,2,1])
        #x3 = tf.strided_slice(x, [0,1,1,0],[128,16,16,64],[1,2,2,1])
        x0 = x[:, :, 0:1, :, 0:1, :]
        x1 = x[:, :, 1:2, :, 0:1, :]
        x2 = x[:, :, 0:1, :, 1:2, :]
        x3 = x[:, :, 1:2, :, 1:2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)


"""
### Build the model

We put together the Swin Transformer model.
"""
distribution_strategy = tf.distribute.OneDeviceStrategy("device:GPU:0")
print('='*20, 'Train', '='*20)
with distribution_strategy.scope():

    input = layers.Input(input_shape)
    print(f'input:{input.shape}')
    # 此处因输入图片尺寸和裁剪尺寸相同，均为(32,32)，裁剪步骤亦可跳过。
    x = my_random_crop(input,(image_dimension, image_dimension))
    #x = layers.RandomCrop(image_dimension, image_dimension)(input, training=True)
    print(f'x-shape:{x.shape}')
    x = layers.RandomFlip("horizontal")(x, training=True)
    x = PatchExtract(patch_size)(x)
    print(f'x-shape:{x.shape}')
    x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    print(f'SwinTransformer-x.shape:{x.shape}')
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    print(f'SwinTransformer-x.shape:{x.shape}')
    x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    """
    ### Train on CIFAR-100

    We train the model on CIFAR-100. Here, we only train the model
    for 40 epochs to keep the training time short in this example.
    In practice, you should train for 150 epochs to reach convergence.
    """


    model = keras.Model(input, output)

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    #optimizer = npu_device.train.optimizer.NpuLossScaleOptimizer(optimizer,dynamic=False,initial_scale=1)
    opt = npu_device.distribute.npu_distributed_keras_optimizer_wrapper(optimizer)
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        optimizer=opt,
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    training_vars = model.trainable_variables
    npu_device.distribute.broadcast(training_vars, root_rank = 0)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    #shuffle=True,
    epochs=num_epochs,
    validation_split=validation_split,
    verbose=1,
)

"""
Let's visualize the training progress of the model.
"""

#plt.plot(history.history["loss"], label="train_loss")
#plt.plot(history.history["val_loss"], label="val_loss")
#plt.xlabel("Epochs")
#plt.ylabel("Loss")
#plt.title("Train and Validation Losses Over Epochs", fontsize=14)
#plt.legend()
#plt.grid()
#plt.show()

"""
Let's display the final results of the training on CIFAR-100.
"""

print('='*20, 'Evaluate', '='*20)
loss, accuracy, top_5_accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test loss: {round(loss, 2)}")
print(f"Test accuracy: {round(accuracy * 100, 2)}%")
print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

"""
The Swin Transformer model we just trained has just 152K parameters, and it gets
us to ~75% test top-5 accuracy within just 40 epochs without any signs of overfitting
as well as seen in above graph. This means we can train this network for longer
(perhaps with a bit more regularization) and obtain even better performance.
This performance can further be improved by additional techniques like cosine
decay learning rate schedule, other data augmentation techniques. While experimenting,
I tried training the model for 150 epochs with a slightly higher dropout and greater
embedding dimensions which pushes the performance to ~72% test accuracy on CIFAR-100
as you can see in the screenshot.

![Results of training for longer](https://i.imgur.com/9vnQesZ.png)

The authors present a top-1 accuracy of 87.3% on ImageNet. The authors also present
a number of experiments to study how input sizes, optimizers etc. affect the final
performance of this model. The authors further present using this model for object detection,
semantic segmentation and instance segmentation as well and report competitive results
for these. You are strongly advised to also check out the
[original paper](https://arxiv.org/abs/2103.14030).

This example takes inspiration from the official
[PyTorch](https://github.com/microsoft/Swin-Transformer) and
[TensorFlow](https://github.com/VcampSoldiers/Swin-Transformer-Tensorflow) implementations.
"""
