# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Contains definitions for the post-activation form of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

from six.moves import range
import tensorflow.compat.v1 as tf

from modeling.architecture import nn_blocks
from modeling.architecture import nn_ops


def get_drop_connect_rate(init_rate, block_num, total_blocks):
  """Get drop connect rate for the ith block."""
  if init_rate is not None:
    return init_rate * float(block_num) / total_blocks
  else:
    return None


def block_group(inputs,
                filters,
                strides,
                use_projection,
                block_fn,
                block_repeats,
                activation=tf.nn.relu,
                batch_norm_activation=nn_ops.BatchNormActivation(),
                dropblock=nn_ops.Dropblock(),
                drop_connect_rate=None,
                data_format='channels_last',
                name=None,
                is_training=False):
  """Builds one group of blocks.

  Args:
    inputs: a `Tensor` of size `[batch, channels, height, width]`.
    filters: an `int` number of filters for the first two convolutions.
    strides: an `int` block stride. If greater than 1, this block will
      ultimately downsample the input.
    use_projection: a `bool` for whether this block should use a projection
      shortcut (versus the default identity shortcut). This is usually `True`
      for the first block of a block group, which may change the number of
      filters and the resolution.
    block_fn: the `function` for the block to use within the model
    block_repeats: an `int` number of blocks to repeat in the group.
    activation: activation function. Support 'relu' and 'swish'.
    batch_norm_activation: an operation that includes a batch normalization
      layer followed by an optional activation layer.
    dropblock: a drop block layer that is added after convluations. Note that
      the default implementation does not apply any drop block.
    drop_connect_rate: a 'float' number that specifies the drop connection rate
      of the block. Note that the default `None` means no drop connection is
      applied.
    data_format: a `str` that specifies the data format.
    name: a `str` name for the Tensor output of the block layer.
    is_training: a `bool` if True, the model is in training mode.

  Returns:
    The output `Tensor` of the block layer.
  """
  # Only the first block per block_group uses projection shortcut and strides.
  inputs = block_fn(
      inputs,
      filters,
      strides,
      use_projection=use_projection,
      activation=activation,
      batch_norm_activation=batch_norm_activation,
      dropblock=dropblock,
      drop_connect_rate=drop_connect_rate,
      data_format=data_format,
      is_training=is_training)
  for _ in range(1, block_repeats):
    inputs = block_fn(
        inputs,
        filters,
        1,
        use_projection=False,
        activation=activation,
        batch_norm_activation=batch_norm_activation,
        dropblock=dropblock,
        drop_connect_rate=drop_connect_rate,
        data_format=data_format,
        is_training=is_training)
  return tf.identity(inputs, name)


class Resnet(object):
  """Class to build ResNet family model."""

  def __init__(self,
               resnet_depth,
               dropblock=nn_ops.Dropblock(),
               activation='relu',
               batch_norm_activation=nn_ops.BatchNormActivation(),
               init_drop_connect_rate=None,
               data_format='channels_last',
               space_to_depth_block_size=1):
    """ResNet initialization function.

    Args:
      resnet_depth: `int` depth of ResNet backbone model.
      dropblock: a dropblock layer.
      activation: activation function. Support 'relu' and 'swish'.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
      init_drop_connect_rate: a 'float' number that specifies the initial drop
        connection rate. Note that the default `None` means no drop connection
        is applied.
      data_format: `str` either "channels_first" for `[batch, channels, height,
        width]` or "channels_last for `[batch, height, width, channels]`.
      space_to_depth_block_size: an integer indicates the block size of
        space-to-depth convolution for conv0. `0` means use the original conv2d
        in ResNet
    """
    self._resnet_depth = resnet_depth

    self._dropblock = dropblock
    if activation == 'relu':
      self._activation = tf.nn.relu
    elif activation == 'swish':
      self._activation = tf.nn.swish
    else:
      raise ValueError('Activation {} not implemented.'.format(activation))
    self._batch_norm_activation = batch_norm_activation
    self._init_drop_connect_rate = init_drop_connect_rate

    self._data_format = data_format
    self._space_to_depth_block_size = space_to_depth_block_size

    model_params = {
        10: {'block': nn_blocks.residual_block, 'layers': [1, 1, 1, 1]},
        14: {'block': nn_blocks.bottleneck_block, 'layers': [1, 1, 1, 1]},
        18: {'block': nn_blocks.residual_block, 'layers': [2, 2, 2, 2]},
        26: {'block': nn_blocks.bottleneck_block, 'layers': [2, 2, 2, 2]},
        34: {'block': nn_blocks.residual_block, 'layers': [3, 4, 6, 3]},
        50: {'block': nn_blocks.bottleneck_block, 'layers': [3, 4, 6, 3]},
        101: {'block': nn_blocks.bottleneck_block, 'layers': [3, 4, 23, 3]},
        152: {'block': nn_blocks.bottleneck_block, 'layers': [3, 8, 36, 3]},
        200: {'block': nn_blocks.bottleneck_block, 'layers': [3, 24, 36, 3]}
    }

    if resnet_depth not in model_params:
      valid_resnet_depths = ', '.join(
          [str(depth) for depth in sorted(model_params.keys())])
      raise ValueError(
          'The resnet_depth should be in [%s]. Not a valid resnet_depth:'%(
              valid_resnet_depths), self._resnet_depth)
    params = model_params[resnet_depth]
    self._resnet_fn = self.resnet_v1_generator(params['block'],
                                               params['layers'],
                                               self._space_to_depth_block_size)

  def __call__(self, inputs, is_training=False):
    """Returns the ResNet model for a given size and number of output classes.

    Args:
      inputs: a `Tesnor` with shape [batch_size, height, width, 3] representing
        a batch of images.
      is_training: `bool` if True, the model is in training mode.

    Returns:
      a `dict` containing `int` keys for continuous feature levels [2, 3, 4, 5].
      The values are corresponding feature hierarchy in ResNet with shape
      [batch_size, height_l, width_l, num_filters].
    """
    with tf.variable_scope('resnet%s' % self._resnet_depth):
      return self._resnet_fn(inputs, is_training)

  def resnet_v1_generator(self, block_fn, layers, space_to_depth_block_size=1):
    """Generator for ResNet v1 models.

    Args:
      block_fn: `function` for the block to use within the model. Either
          `residual_block` or `bottleneck_block`.
      layers: list of 4 `int`s denoting the number of blocks to include in each
        of the 4 block groups. Each group consists of blocks that take inputs of
        the same resolution.
      space_to_depth_block_size: an integer indicates the block size of
        space-to-depth convolution for conv0. `0` means use the original conv2d
        in ResNet.

    Returns:
      Model `function` that takes in `inputs` and `is_training` and returns the
      output `Tensor` of the ResNet model.
    """
    def model(inputs, is_training=False):
      """Creation of the model graph."""
      if space_to_depth_block_size > 1 and is_training:
        # conv0 uses space-to-depth transform for TPU performance.
        inputs = nn_ops.conv0_space_to_depth(
            inputs=inputs,
            filters=64,
            kernel_size=7,
            strides=2,
            data_format=self._data_format,
            space_to_depth_block_size=space_to_depth_block_size)
      else:
        inputs = nn_ops.conv2d_fixed_padding(
            inputs=inputs,
            filters=64,
            kernel_size=7,
            strides=2,
            data_format=self._data_format)
      inputs = tf.identity(inputs, 'initial_conv')
      inputs = self._batch_norm_activation(inputs, is_training=is_training)

      inputs = tf.layers.max_pooling2d(
          inputs=inputs, pool_size=3, strides=2, padding='SAME',
          data_format=self._data_format)
      inputs = tf.identity(inputs, 'initial_max_pool')

      c2 = block_group(
          inputs=inputs,
          filters=64,
          strides=1,
          use_projection=True,
          block_fn=block_fn,
          block_repeats=layers[0],
          activation=self._activation,
          batch_norm_activation=self._batch_norm_activation,
          dropblock=self._dropblock,
          drop_connect_rate=get_drop_connect_rate(
              self._init_drop_connect_rate, 2, 5),
          name='block_group1',
          is_training=is_training)
      c3 = block_group(
          inputs=c2,
          filters=128,
          strides=2,
          use_projection=True,
          block_fn=block_fn,
          block_repeats=layers[1],
          activation=self._activation,
          batch_norm_activation=self._batch_norm_activation,
          dropblock=self._dropblock,
          drop_connect_rate=get_drop_connect_rate(
              self._init_drop_connect_rate, 3, 5),
          name='block_group2',
          is_training=is_training)
      c4 = block_group(
          inputs=c3,
          filters=256,
          strides=2,
          use_projection=True,
          block_fn=block_fn,
          block_repeats=layers[2],
          activation=self._activation,
          batch_norm_activation=self._batch_norm_activation,
          dropblock=self._dropblock,
          drop_connect_rate=get_drop_connect_rate(
              self._init_drop_connect_rate, 4, 5),
          name='block_group3',
          is_training=is_training)
      c5 = block_group(
          inputs=c4,
          filters=512,
          strides=2,
          use_projection=True,
          block_fn=block_fn,
          block_repeats=layers[3],
          activation=self._activation,
          batch_norm_activation=self._batch_norm_activation,
          dropblock=self._dropblock,
          drop_connect_rate=get_drop_connect_rate(
              self._init_drop_connect_rate, 5, 5),
          name='block_group4',
          is_training=is_training)
      return {2: c2, 3: c3, 4: c4, 5: c5}

    return model

