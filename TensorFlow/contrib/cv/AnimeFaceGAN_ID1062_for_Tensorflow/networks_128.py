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
from ops import *


class Generator:
    def __init__(self, name, base_channel):
        self.name = name
        self.base_channel = base_channel

    def __call__(self, inputs, train_phase, y, embed_w, nums_class, y_end=None, alpha=1.0):
        """
        Args:
            inputs: a noise vector. shape: [batch_size, z_dim]
            train_phase: is training or not
            y: class info
            embed_w: weight for shared embedding
            nums_class: number of image classes
        """
        # hierarchical latent space: split z into one chunk per resolution
        z_dim = int(inputs.shape[-1])
        nums_layer = 6
        remain = z_dim % nums_layer
        chunk_size = (z_dim - remain) // nums_layer
        z_split = tf.split(inputs, [chunk_size] * (nums_layer - 1) + [chunk_size + remain], axis=1)
        y = tf.one_hot(y, nums_class)

        if not y_end is None:
            # category morphing
            y_end = tf.one_hot(y_end, nums_class)
            y = y * alpha + y_end * (1 - alpha)

        embed_y = tf.matmul(y, embed_w)  # shared embedding
        inputs = tf.concat([z_split[0], embed_y], axis=1)

        ch = self.base_channel     # base channel number per layer
        out_channels = [ch * i for i in [16, 8, 4, 2, 1]]

        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            inputs = dense("dense", inputs, 4*4*out_channels[0])
            inputs = tf.reshape(inputs, [-1, 4, 4, out_channels[0]])
            inputs = G_Resblock("ResBlock1", inputs, out_channels[0], train_phase, z_split[1], embed_y)
            inputs = G_Resblock("ResBlock2", inputs, out_channels[1], train_phase, z_split[2], embed_y)
            inputs = G_Resblock("ResBlock3", inputs, out_channels[2], train_phase, z_split[3], embed_y)
            inputs = G_Resblock("ResBlock4", inputs, out_channels[3], train_phase, z_split[4], embed_y)
            inputs = non_local("Non-local", inputs, None, is_sn=True)
            inputs = G_Resblock("ResBlock5", inputs, out_channels[4], train_phase, z_split[5], embed_y)
            inputs = tf.nn.relu(conditional_batchnorm(inputs, train_phase, "BN"))   # batch normalization
            inputs = conv("conv", inputs, k_size=3, nums_out=3, strides=1, is_sn=True)
        return tf.nn.tanh(inputs)

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


class Discriminator:
    def __init__(self, name, base_channel):
        self.name = name
        self.base_channel = base_channel

    def __call__(self, inputs, train_phase, y, nums_class, update_collection=None):
        """
        Args:
            inputs: an image. shape: [batch_size, 128, 128, 3]
            y: class info (scalar)
            nums_class: number of image classes
        """
        ch = self.base_channel  # base channel number per layer
        out_channels = [ch * i for i in [1, 2, 4, 8, 16, 16]]

        with tf.variable_scope(name_or_scope=self.name, reuse=tf.AUTO_REUSE):
            inputs = D_Resblock("ResBlock1", inputs, out_channels[0], train_phase, update_collection, is_down=True)      # [N, 64, 64, ch]
            inputs = non_local("Non-local", inputs, update_collection, True)
            inputs = D_Resblock("ResBlock2", inputs, out_channels[1], train_phase, update_collection, is_down=True)    # [N, 32, 32, 2*ch]
            inputs = D_Resblock("ResBlock3", inputs, out_channels[2], train_phase, update_collection, is_down=True)   # [N, 16, 16, 4*ch]
            inputs = D_Resblock("ResBlock4", inputs, out_channels[3], train_phase, update_collection, is_down=True)   # [N, 8, 8, 8*ch]
            inputs = D_Resblock("ResBlock5", inputs, out_channels[4], train_phase, update_collection, is_down=True)  # [N, 4, 4, 16*ch]
            inputs = D_Resblock("ResBlock6", inputs, out_channels[5], train_phase, update_collection, is_down=False)
            inputs = tf.nn.relu(inputs)
            inputs = global_sum_pooling(inputs)         # [N, 16*ch]
            temp = d_projection(inputs, y, nums_class, update_collection)       # [N, 1]
            inputs = dense("dense", inputs, 1, update_collection, is_sn=True)   # [N, 1]
            inputs = temp + inputs
            return inputs

    def var_list(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

