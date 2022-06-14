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

# from npu_bridge.npu_init import *
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np

def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    return resnet34(images,phase_train),None

def bn(x,is_training):
    out = tf.layers.batch_normalization(x, training=True,momentum=0.995,epsilon=0.001)
    return out

def conv2d(x,out_channels,kernel_size,strides,padding,activation,useNormalize,is_training):
    out=tf.layers.conv2d(inputs=x,filters=out_channels,kernel_size=kernel_size,strides=strides,
                         padding=padding)
    if useNormalize:
        out=bn(out,is_training)
    if activation:
        out=activation(out)
    return out

def BasicBlock(x,is_training,in_channels,out_channels,strides=1):
    out = conv2d(x, out_channels, [3, 3], [strides, strides], padding="same", activation=tf.nn.relu,
                 useNormalize=True,is_training=is_training)
    out = conv2d(out, out_channels, [3, 3], [1, 1], padding="same", activation=None,
                 useNormalize=True,is_training=is_training)
    if strides!=1 or in_channels!=out_channels:
        x=conv2d(x,out_channels,[1,1],[strides,strides],padding="same",activation=None,
                 useNormalize=True,is_training=is_training)
    out+=x
    return tf.nn.relu(out)

def stage(x,is_training,block,in_channels,out_channels,num_blocks,stride):
    strides=[stride]+[1]*(num_blocks-1)
    for i in range(len(strides)):
        if i==0:
            out=block(x,is_training,in_channels,out_channels,strides[i])
        else:
            out=block(out,is_training,in_channels,out_channels,strides[i])
        in_channels=out_channels
    return out

def resnet34(x,is_training):
    out = conv2d(x, 64, [7, 7], [2, 2], padding="same", activation=tf.nn.relu,
                 useNormalize=True,is_training=is_training)
    #ResNet(BasicBlock, [3,4,6,3])
    out = stage(out,is_training,BasicBlock,64,64,3,1)
    out = stage(out,is_training,BasicBlock,64,128,4,2)
    out = stage(out,is_training,BasicBlock,128,256,6,2)
    out = stage(out,is_training,BasicBlock,256,512,3,2)
    out = tf.reduce_mean(out,[1,2],keepdims=True)
    size = np.prod(out.get_shape()[1:].as_list())
    out=tf.reshape(out, (-1, size))
    return out



