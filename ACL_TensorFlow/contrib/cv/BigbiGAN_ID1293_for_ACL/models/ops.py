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
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import Wrapper
import tensorflow_addons
from tensorflow_addons.layers import SpectralNormalization

weight_regularizer = None #orthogonal_regularizer(0.0001)
weight_regularizer_fully = None

class Attention(tf.keras.Model):
    'https://stackoverflow.com/questions/50819931/self-attention-gan-in-keras'
    def __init__(self, ch, config):
        super(Attention, self).__init__()
        self.filters_f_g_h = ch // 8
        self.filters_v = ch
        self.f_l = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.filters_f_g_h, kernel_size=1, strides=1, use_bias=True))
        self.g_l = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.filters_f_g_h, kernel_size=1, strides=1, use_bias=True))
        self.h_l = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.filters_f_g_h, kernel_size=1, strides=1, use_bias=True))
        self.v_l = SpectralNormalization(tf.keras.layers.Conv2D(filters=self.filters_v, kernel_size=1, strides=1, use_bias=True))
        self.gamma = tf.Variable(initial_value=[0], dtype=tf.float32, trainable=True)

    def __call__(self, inputs, training=False):
        def hw_flatten(x):
            return tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1]*tf.shape(x)[2], tf.shape(x)[-1]])

        f = self.f_l(inputs)

        g = self.g_l(inputs)

        h = self.h_l(inputs)

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        v = tf.matmul(beta, hw_flatten(h))
        v = tf.reshape(v, shape=[tf.shape(inputs)[0],tf.shape(inputs)[1],tf.shape(inputs)[2], -1]) # [bs, h, w, C]

        o = self.v_l(v)

        output = self.gamma * o + inputs

        return output


class resblock(tf.keras.Model):
    def __init__(self, channels, config, weight_init, use_bias=True):
        super().__init__()
        with tf.name_scope('resblock'):
            self.conv0 = SpectralNormalization(Conv2D(filters=channels,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='SAME',
                                                                use_bias=use_bias,
                                                                kernel_initializer=weight_init))
            self.conv1 = SpectralNormalization(Conv2D(filters=channels,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='SAME',
                                                                use_bias=use_bias,
                                                                kernel_initializer=weight_init))

    def __call__(self, inputs, training=False):
        # res1
        x = self.conv0(inputs)
        x = tf.nn.relu(x)
        # res2
        x = self.conv1(x)

        return x+inputs

class resblock_up_condition_top(tf.keras.Model):

    def __init__(self, channels, config, weight_init, use_bias=True):
        super().__init__()
        with tf.name_scope('resblock_up_condition_top'):
            self.cond_batchnorm0 = condition_batch_norm(channels=channels, weight_init=weight_init)
            self.upsampling = tf.keras.layers.UpSampling2D()
            self.conv1 = SpectralNormalization(Conv2D(filters=channels,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='SAME',
                                                                use_bias=use_bias,
                                                                kernel_initializer= weight_init,
                                                                kernel_regularizer= weight_regularizer))
            self.cond_batchnorm1 = condition_batch_norm(channels=channels, weight_init=weight_init)
            self.conv2 = SpectralNormalization(Conv2D(filters=channels,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='SAME',
                                                                use_bias=use_bias,
                                                                kernel_initializer= weight_init,
                                                                kernel_regularizer= weight_regularizer))
            self.skip_conv = SpectralNormalization(Conv2D(filters=channels,
                                                                   kernel_size=1,
                                                                   strides=1,
                                                                   padding='SAME',
                                                                   use_bias=use_bias,
                                                                   kernel_initializer= weight_init,
                                                                   kernel_regularizer= weight_regularizer))

    def __call__(self, inputs, z, training=False):
        # res1
        x = self.cond_batchnorm0(inputs, z, training=training)
        x = tf.nn.relu(x)
        x = self.upsampling(x)
        x = self.conv1(x)

        # res2
        x = self.cond_batchnorm1(x, z, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        # skip
        x_init = self.upsampling(inputs)
        x_init = self.skip_conv(x_init)

        return x + x_init

class resblock_up_condition(tf.keras.Model):

    def __init__(self, channels, config, weight_init, use_bias=True):
        super().__init__()
        with tf.name_scope('resblock_up_condition'):
            self.cond_batchnorm0 = condition_batch_norm(channels=channels * 2, weight_init=weight_init)
            self.upsampling = tf.keras.layers.UpSampling2D()
            self.conv1 = SpectralNormalization(Conv2D(filters=channels,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='SAME',
                                                                use_bias=use_bias,
                                                                kernel_initializer= weight_init,
                                                                kernel_regularizer= weight_regularizer))
            self.cond_batchnorm1 = condition_batch_norm(channels=channels, weight_init=weight_init)
            self.conv2 = SpectralNormalization(Conv2D(filters=channels,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='SAME',
                                                                use_bias=use_bias,
                                                                kernel_initializer= weight_init,
                                                                kernel_regularizer= weight_regularizer))
            self.skip_conv = SpectralNormalization(Conv2D(filters=channels,
                                                                   kernel_size=1,
                                                                   strides=1,
                                                                   padding='SAME',
                                                                   use_bias=use_bias,
                                                                   kernel_initializer= weight_init,
                                                                   kernel_regularizer= weight_regularizer))

    def __call__(self, inputs, z, training=False):
        # res1
        x = self.cond_batchnorm0(inputs, z, training=training)
        x = tf.nn.relu(x)
        x = self.upsampling(x)
        x = self.conv1(x)

        # res2
        x = self.cond_batchnorm1(x, z, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        # skip
        x_init = self.upsampling(inputs)
        x_init = self.skip_conv(x_init)

        return x + x_init

class resblock_down(tf.keras.Model):
    def __init__(self, channels, config, weight_init, use_bias=True):
        super().__init__()
        with tf.name_scope('resblock_down'):

            self.conv0 = SpectralNormalization(Conv2D(filters=channels,
                                                                kernel_size=3,
                                                                strides=1,
                                                                padding='SAME',
                                                                use_bias=use_bias,
                                                                kernel_initializer=weight_init))
            self.conv1 = SpectralNormalization(Conv2D(filters=channels,
                                                      kernel_size=3,
                                                      strides=1,
                                                      padding='SAME',
                                                      use_bias=use_bias,
                                                      kernel_initializer=weight_init))
            self.skip_conv = SpectralNormalization(Conv2D(filters=channels,
                                                      kernel_size=1,
                                                      strides=1,
                                                      padding='SAME',
                                                      use_bias=use_bias,
                                                      kernel_initializer=weight_init))
            self.avg_pooling = tf.keras.layers.AveragePooling2D(padding='SAME')

    def __call__(self, inputs, training=False):
        # res1
        x = tf.nn.relu(inputs)
        x = self.conv0(x)
        # res2
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.avg_pooling(x)
        # skip
        x_init = self.skip_conv(inputs)
        x_init = self.avg_pooling(x_init)

        return x + x_init


class resblock_dense(tf.keras.Model):
    def __init__(self, units, weight_init, config):
        super().__init__()
        with tf.name_scope('resblock_dense'):
            self.dense0 = SpectralNormalization(Dense(units=units, kernel_initializer= weight_init))
            self.dropout = tf.keras.layers.Dropout(0.2)
            self.dense1 = SpectralNormalization(Dense(units=units, kernel_initializer= weight_init))
            self.dense_skip = SpectralNormalization(Dense(units=units, kernel_initializer=weight_init))

    #@tf.function
    def __call__(self, inputs, training=False):

        l1 = self.dense0(inputs)
        l1 = self.dropout(l1, training=training)
        l1 = tf.nn.leaky_relu(l1)

        l2 = self.dense1(l1)
        l2 = self.dropout(l2, training=training)
        l2 = tf.nn.leaky_relu(l2)

        skip = self.dense_skip(inputs)
        skip = tf.nn.leaky_relu(skip)


        output = l2+skip

        return output


class resblock_dense_no_sn(tf.keras.Model):
    def __init__(self, units, weight_init):
        super().__init__()
        with tf.name_scope('resblock_dense_no_sn'):
            self.dense0 = Dense(units=units, kernel_initializer=weight_init)
            self.dropout = tf.keras.layers.Dropout(0.2)
            self.dense1 = Dense(units=units, kernel_initializer=weight_init)
            self.dense_skip = Dense(units=units, kernel_initializer=weight_init)

    def __call__(self, inputs, training=False):

        l1 = self.dense0(inputs)
        l1 = self.dropout(l1, training=training)
        l1 = tf.nn.leaky_relu(l1)

        l2 = self.dense1(l1)
        l2 = self.dropout(l2, training=training)
        l2 = tf.nn.leaky_relu(l2)

        skip = self.dense_skip(inputs)
        skip = tf.nn.leaky_relu(skip)

        output = l2+skip

        return output



class bottleneck_s(tf.keras.Model):
    def __init__(self, filters, weight_init, strides=1):
        super().__init__()
        self.filters = filters
        self.strides = strides
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.conv0 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='SAME', use_bias=False, kernel_initializer=weight_init)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='SAME', strides=strides, use_bias=False,kernel_initializer=weight_init)
        self.skip_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=strides, padding='SAME', kernel_initializer=weight_init)

    def __call__(self, inputs, training=False):
        l1 = self.conv0(inputs)
        l1 = self.bn0(l1, training=training)
        l1 = tf.nn.relu(l1)

        l2 = self.conv1(l1)
        l2 = self.bn1(l2, training=training)
        l2 = tf.nn.relu(l2)

        # Project input if necessary
        if (self.strides > 1) or (self.filters != inputs.get_shape().as_list()[-1]):
            x_shortcut = self.skip_conv(inputs)
        else:
            x_shortcut = inputs

        return l2 + x_shortcut

class bottleneck_rev_s(tf.keras.Model):
    def __init__(self, ch, weight_init, strides=1):
        super().__init__()
        self.unit = bottleneck_s(filters=ch//2, strides=strides, weight_init=weight_init)

    def __call__(self, inputs, training=False):
        # split with 2 parts and along axis=3
        x1, x2 = tf.split(inputs, 2, 3)
        y1 = x1 + self.unit(x2, training=training)
        y2 = x2

        # concatenate y2 and y1 along axis=3
        return tf.concat([y2, y1], axis=3)

def pool_and_double_channels(x, pool_stride):

    if pool_stride > 1:
        x = tf.nn.avg_pool2d(x, ksize=pool_stride,
                                        strides=pool_stride,
                                        padding='SAME')
    return tf.pad(x, [[0, 0], [0, 0], [0, 0],
                      [x.get_shape().as_list()[3] // 2, x.get_shape().as_list()[3] // 2]])



class condition_batch_norm(tf.keras.Model):
    def __init__(self, channels, weight_init):
        super().__init__()
        self.channels = channels
        self.decay = 0.9
        self.epsilon = 1e-05
        self.test_mean = tf.Variable(tf.zeros([channels]), dtype=tf.float32, trainable=False)
        self.test_var = tf.Variable(tf.ones([channels]), dtype=tf.float32, trainable=False)
        self.beta0 = tf.keras.layers.Dense(units=channels, use_bias=True, kernel_initializer=weight_init, kernel_regularizer= weight_regularizer_fully)
        self.gamma0 = tf.keras.layers.Dense(units=channels, use_bias=True, kernel_initializer=weight_init, kernel_regularizer= weight_regularizer_fully)

    def __call__(self, x, z, training=False):

        beta0 = self.beta0(z)

        gamma0 = self.gamma0(z)

        beta = tf.reshape(beta0, shape=[-1, 1, 1, self.channels])
        gamma = tf.reshape(gamma0, shape=[-1, 1, 1, self.channels])

        if training:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            self.test_mean.assign(self.test_mean * self.decay + batch_mean * (1 - self.decay))
            self.test_var.assign(self.test_var * self.decay + batch_var * (1 - self.decay))

            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, self.epsilon)
        else:
            return tf.nn.batch_normalization(x, self.test_mean, self.test_var, beta, gamma, self.epsilon)



