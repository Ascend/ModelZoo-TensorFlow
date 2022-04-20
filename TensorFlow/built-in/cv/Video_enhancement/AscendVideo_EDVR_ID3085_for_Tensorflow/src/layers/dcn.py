# Copyright 2022 Huawei Technologies Co., Ltd
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

import math

import tensorflow as tf
import numpy as np

from .conv import Conv2D
from .base_layer import BaseLayer

from src.utils.utils import to_pair
from src.utils.logger import logger


try:
    from npu_bridge.tbe.npu_cube_ops import deformable_conv2d
    OP_IMPL = 'npu'
except Exception:
    logger.error('Failed to import NPU deformable_conv2d. '
                 'Please use the composed tf operator instead.'
                 '(This is NOT an actual error)')
    OP_IMPL = 'tf'



__all__ = ["DCNPack"]

class DeformableConvLayer(BaseLayer):
    """Deformable convolution layer.

    Args:
        in_channels: int, number of channels of the input feature.
        out_channels: int, number of channels of the output feature.
        kernel_size: int or list[int] or tuple[int], kernel size of the conv
            operation.
        strides: int or list[int] or tuple[int], strides of the conv.
        padding: str, options in ['same', 'valid']. Case insensitive.
        dilations: int or list[int] or tuple[int], dilations of the conv.
        use_bias: boolean, whether to add bias or not. Default True.
        num_groups: int, number of convolution groups.
        num_deform_groups: int, number of the groups of the offsets.
        trainable: boolean, whether to train the parameters.
        impl: str, which operator to use. Options in ['tf', 'npu']. If using
            'tf' version, the DCN will be composed of the tensorflow operators,
            which may be memory and runtime inefficient. For Ascned platform,
            we recommend to use npu deformable_conv2d instead.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilations=1,
                 use_bias=True,
                 num_groups=1,
                 num_deform_groups=1,
                 trainable=True,
                 impl='tf'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_pair(kernel_size, 2)
        self.strides = to_pair(strides, 2)
        self.padding = padding.lower()
        self.dilations = to_pair(dilations, 2)
        self.use_bias = use_bias
        self.num_groups = num_groups
        self.num_deform_groups = num_deform_groups
        self.trainable = trainable
        self.kernel_intermediate_shape = []
        self.build()
        self.impl = impl
        
    def build(self):
        """Prepare the weights and bias.
        """
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        initializer = tf.random_uniform_initializer(-stdv, stdv)

        self.kernel_intermediate_shape = [*self.kernel_size, self.in_channels//self.num_groups, self.out_channels//self.num_groups, self.num_groups]

        self.kernel = tf.get_variable(
            "W",
            [*self.kernel_size, self.in_channels//self.num_groups, self.out_channels],
            initializer=initializer,
            trainable=self.trainable)
        if self.use_bias:
            self.bias = tf.get_variable(
                "b",
                (self.out_channels,),
                initializer=tf.constant_initializer(value=0.0),
                trainable=self.trainable)
    
    def _cal_pads(self, ih, iw):
        """Calculation padding given the input.
        """
        if self.padding == 'same':
            strh, strw = self.strides
            kh, kw = self.kernel_size
            dilh, dilw = self.dilations
            tails_h = ih % strh
            tails_w = iw % strw
            dkh = dilh * (kh - 1) + 1
            dkw = dilw * (kw - 1) + 1
            pad_h = dkh - tails_h if tails_h > 0 else dkh - strh
            pad_w = dkw - tails_w if tails_w > 0 else dkw - strw
            pads = [pad_h // 2, pad_h // 2 + pad_h % 2, pad_w // 2, pad_w // 2 + pad_w % 2]
        else:
            pads = [0, 0, 0, 0]
        return pads   
 
    def forward(self, inputs, offset):
        """Deformable Conv2d forward function.
        """
        if self.impl == 'npu' and OP_IMPL == 'npu':
            return self._forward_npu(inputs, offset)
        else:
            return self._forward_tf(inputs, offset)

    def _forward_npu(self, inputs, offset):
        """Forward function of NPU deformable operator.
        """
        _, ih, iw, _ = inputs.get_shape().as_list()
        c = offset.get_shape().as_list()[3]
        assert c == self.num_deform_groups*self.kernel_size[0]*self.kernel_size[1]*3
        offset_all = offset

        pads = self._cal_pads(ih, iw)
        out = deformable_conv2d(
                inputs,
                self.kernel,
                offset_all,
                strides=[1] + list(self.strides) + [1],
                pads=pads,
                data_format='NHWC',
                dilations=[1] + list(self.dilations) + [1],
                groups=self.num_groups,
                deformable_groups=self.num_deform_groups)

        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias)
        return out

    def _forward_tf(self, inputs, offset):
        """Forward function of tf composed deformable operator.
        """
        def _get_in_bound_mask(x_, y_):
            out_of_bound_x = tf.logical_or(tf.greater(x_, in_w-1), tf.less(x_, 0))
            out_of_bound_y = tf.logical_or(tf.greater(y_, in_h-1), tf.less(y_, 0))
            out_of_bound_mask = tf.logical_or(out_of_bound_x, out_of_bound_y)
            return 1. - tf.to_float(out_of_bound_mask)

        inputs = self._pad_input(inputs)
        bs, in_h, in_w, _ = inputs.get_shape().as_list()
        bs, out_h, out_w, c = offset.get_shape().as_list()

        assert c == self.num_deform_groups*self.kernel_size[0]*self.kernel_size[1]*3
        c3 = c // 3

        # get x, y axis offset. Swap the order to 'x,y' instead of 'y,x', align with npu dcn op
        x_off = offset[:, :, :, :c3]
        y_off = offset[:, :, :, c3:c3*2]
        mask = offset[:, :, :, c3*2:]
        
        # input feature map gird coordinates
        y, x = self._get_conv_indices(in_h, in_w)
        y, x = [tf.to_float(i) for i in [y, x]]
        y, x = [tf.tile(i, [1, 1, 1, self.num_deform_groups]) for i in [y, x]]
        
        # current deformable offsets
        y, x = y + y_off, x + x_off

        # get four coordinates of points around (x, y)
        y0, x0 = [tf.to_int32(tf.floor(i)) for i in [y, x]]
        y1, x1 = y0 + 1, x0 + 1
        
        # according to the strategy, prepare in_bound mask if use zero.
        # In fact, gathernd NPU will take 0 if the index is out-of-bound,
        # while CPU will throw an error. Therefore, do an explicit masking
        m0 = _get_in_bound_mask(x0, y0)
        m1 = _get_in_bound_mask(x1, y0)
        m2 = _get_in_bound_mask(x0, y1)
        m3 = _get_in_bound_mask(x1, y1)

        y_res = y - tf.to_float(y0)
        x_res = x - tf.to_float(x0)

        w0_ori = (1. - y_res) * (1. - x_res)
        w1_ori = (1. - y_res) * x_res
        w2_ori = y_res * (1. - x_res)
        w3_ori = y_res * x_res
        
        # clip the indices
        y0_clip, y_clip, y1_clip = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y, y1]]
        x0_clip, x_clip, x1_clip = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x, x1]]
        
        # get pixel values
        indices = [[y0_clip, x0_clip], [y0_clip, x1_clip], [y1_clip, x0_clip], [y1_clip, x1_clip]]
        p0, p1, p2, p3 = [self._get_pixel_values_at_point(inputs, i) for i in indices]
        
        # cast to float
        x0_clip, x_clip, x1_clip, y0_clip, y_clip, y1_clip = [tf.to_float(i) for i in
                                                              [x0_clip, x_clip, x1_clip, y0_clip, y_clip, y1_clip]]
        
        # weights
        w0 = m0 * w0_ori
        w1 = m1 * w1_ori
        w2 = m2 * w2_ori
        w3 = m3 * w3_ori

        w0, w1, w2, w3 = [tf.reshape(i, [*i.get_shape()[:3], self.num_deform_groups, *self.kernel_size, 1])
                          for i in [w0, w1, w2, w3]]

        # bilinear interpolation
        pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])

        if mask is not None:
            pixels = tf.reshape(mask, [*mask.get_shape()[:3], self.num_deform_groups, *self.kernel_size, 1]) * pixels

        # reshape the "big" feature map
        pixels = tf.transpose(pixels, [0,1,4,2,5,3,6])
        pixels = tf.reshape(pixels, [bs, out_h*self.kernel_size[0], out_w*self.kernel_size[1], -1])

        # conv
        kernel_reshaped = tf.reshape(self.kernel, self.kernel_intermediate_shape)
        ich = pixels.shape[-1] // self.num_groups
        out = tf.concat([tf.nn.conv2d(
                pixels[:, :, :, i*ich:(i+1)*ich], 
                kernel_reshaped[:, :, :, :, i], 
                strides=self.kernel_size, 
                padding='VALID',
                )
                for i in range(self.num_groups)], axis=-1)
                
        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias)

        return out

    def _pad_input(self, x):
        """Pad the input before calculating the offsets.
        """
        if self.padding == 'same':
            ih, iw = x.get_shape().as_list()[1:3]
            pads = self._cal_pads(ih, iw)

            if pads[0] + pads[1] + pads[2] + pads[3] != 0:
                x = tf.pad(x, [[0, 0]] + [pads[:2]] + [pads[2:]] + [[0, 0]])

        return x

    def _get_conv_indices(self, feat_h, feat_w):
        """Get the x, y coordinates in the window when a filter sliding on the 
        feature map
        """

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
        x, y = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y]]  # shape [1, h, w, 1]
        x, y = [tf.image.extract_image_patches(i,
                                               [1, *self.kernel_size, 1],
                                               [1, *self.strides, 1],
                                               [1, *self.dilations, 1],
                                               'VALID')
                for i in [x, y]]  # shape [1, out_h, out_w, filter_h * filter_w]
        return y, x

    def _get_pixel_values_at_point(self, inputs, indices):
        """Get pixel values at the given point.
        """
        y, x = indices
        bs, h, w, c = y.get_shape().as_list()[0: 4]

        if c % self.num_deform_groups != 0 or inputs.shape[-1] % self.num_deform_groups != 0:
            raise ValueError

        per_group_offset_ch = c // self.num_deform_groups  # kh*kw
        per_group_input_ch = inputs.shape[-1] // self.num_deform_groups
        batch_idx = tf.reshape(tf.range(0, bs), (bs, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, per_group_offset_ch))
        
        outs = []
        for j in range(self.num_deform_groups):
            pixel_idx = tf.stack([b, y[:, :, :, j*per_group_offset_ch:(j+1)*per_group_offset_ch],
                                  x[:, :, :, j*per_group_offset_ch:(j+1)*per_group_offset_ch]], axis=-1)  # [bs, h, w, per_group_offset_ch, 3]
            outs.append(tf.gather_nd(inputs[:, :, :, j*per_group_input_ch:(j+1)*per_group_input_ch], pixel_idx))  
        outs = tf.concat(outs, axis=-1)  # [bs, h, w, per_group_offset_ch, cin]

        # reshape and transpose the outputs in order to align with the outer axis order
        outs = tf.reshape(outs, [*outs.shape[:3], *self.kernel_size, self.num_deform_groups, -1])
        return tf.transpose(outs, [0,1,2,5,3,4,6])


class DCNPack:
    def __init__(self, 
                 out_channels, 
                 kernel_size=(3, 3), 
                 strides=(1, 1), 
                 padding='same', 
                 dilations=(1, 1),
                 use_bias=True, 
                 num_groups=1, 
                 num_deform_groups=1,  
                 name='DCN',
                 impl='npu'):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilations = dilations
        self.use_bias = use_bias
        self.num_groups = num_groups
        self.num_deform_groups = num_deform_groups
        self.name = name
        self.impl = impl
        
    def __call__(self, x, extra_feat):
        with tf.variable_scope(self.name):
            x = tf.cast(x, tf.float32)

            n_elem = (self.num_deform_groups
                      * self.kernel_size[0]
                      * self.kernel_size[1])

            num_offset_channels = n_elem * 3

            conv_offset = Conv2D(num_offset_channels,
                                 kernel_size=self.kernel_size, 
                                 strides=self.strides, 
                                 padding=self.padding,
                                 dilations=self.dilations,
                                 use_bias=self.use_bias, 
                                 name='conv_offset')(extra_feat)

            conv_offset = tf.cast(conv_offset, tf.float32)

            # Get the modulation
            modulation = tf.nn.sigmoid(conv_offset)
            offset = conv_offset

            # Prepare a masking 
            weight = np.ones((1, 1, 1, num_offset_channels)).astype(np.float32)
            weight[..., n_elem*2:] = 0.
            weight = tf.convert_to_tensor(weight)

            # Make the n_elem*2 channels the offsets, the last n_elem channels
            # the modulation.
            input_offset_mask = weight * offset + (1. - weight) * modulation

            out = DeformableConvLayer(
                in_channels=int(x.shape[-1]), 
                out_channels=self.out_channels,
                kernel_size=self.kernel_size, 
                strides=self.strides, 
                padding=self.padding,
                dilations=self.dilations, 
                use_bias=self.use_bias, 
                num_groups=self.num_groups,
                num_deform_groups=self.num_deform_groups, 
                impl=self.impl)(x, input_offset_mask)

            return out
