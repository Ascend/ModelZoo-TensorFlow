# MIT License

# Copyright (c) 2018 Deniz Engin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
"""
Generator
"""
from npu_bridge.npu_init import *
import tensorflow as tf
import ops
import utils


class Generator:
    """
    generator
    """
    def __init__(self,
                 name,
                 is_training,
                 ngf=64,
                 norm='instance',
                 image_size1=128,
                 image_size2=128):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.norm = norm
        self.is_training = is_training
        self.image_size1 = image_size1
        self.image_size2 = image_size2

    def __call__(self, input):
        """
        Args:
            input: batch_size x width x height x 3
        Returns:
            output: same size as input
        """
        with tf.variable_scope(self.name):
            # conv layers
            c7s1_32 = ops.c7s1_k(input,
                                 self.ngf,
                                 is_training=self.is_training,
                                 norm=self.norm,
                                 reuse=self.reuse,
                                 name='c7s1_32')  # (?, w, h, 32)
            d64 = ops.dk(c7s1_32,
                         2 * self.ngf,
                         is_training=self.is_training,
                         norm=self.norm,
                         reuse=self.reuse,
                         name='d64')  # (?, w/2, h/2, 64)
            d128 = ops.dk(d64,
                          4 * self.ngf,
                          is_training=self.is_training,
                          norm=self.norm,
                          reuse=self.reuse,
                          name='d128')  # (?, w/4, h/4, 128)

            if self.image_size1 <= 128:
                # use 6 residual blocks for 128x128 images
                res_output = ops.n_res_blocks(d128, reuse=self.reuse,
                                              n=6)  # (?, w/4, h/4, 128)
            else:
                # 9 blocks for higher resolution
                res_output = ops.n_res_blocks(d128, reuse=self.reuse,
                                              n=9)  # (?, w/4, h/4, 128)

            # fractional-strided convolution
            u64 = ops.uk(res_output,
                         2 * self.ngf,
                         is_training=self.is_training,
                         norm=self.norm,
                         reuse=self.reuse,
                         name='u64')  # (?, w/2, h/2, 64)
            u32 = ops.uk(u64,
                         self.ngf,
                         is_training=self.is_training,
                         norm=self.norm,
                         reuse=self.reuse,
                         name='u32')  # (?, w, h, 32)

            # conv layer
            # Note: the paper said that ReLU and _norm were used
            # but actually tanh was used and no _norm here
            output = ops.c7s1_k(u32,
                                3,
                                norm=None,
                                activation='tanh',
                                reuse=self.reuse,
                                name='output')  # (?, w, h, 3)

        # set reuse=True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=self.name)

        return output

    def sample(self, input):
        '''
        convert input into int and encode jpeg, return
        '''
        image = utils.batch_convert2int(self.__call__(input))
        image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
        return image
