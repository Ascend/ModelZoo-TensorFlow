# ============================================================================
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2


def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost.  See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(inputs=inputs, axis=1 if data_format == 'channels_first' else -1,
                                         momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                                         scale=True, training=training, fused=True)


def vgg_block(inputs, filters, num_convs, training, kernel_size, maxpool, data_format):
    for i in range(num_convs):
        inputs = batch_norm(tf.layers.conv2d(inputs, filters, kernel_size, 1,
                                             padding='same', activation=tf.nn.relu,
                                             kernel_initializer=tf.glorot_uniform_initializer(),
                                             data_format=data_format), training=training, data_format=data_format)
    if maxpool:
        inputs = tf.layers.max_pooling2d(inputs, 2, 2)

    return inputs


class Model(object):

    def __init__(self,
                 num_lmark,
                 img_size,
                 filter_sizes,
                 num_convs,
                 kernel_size,
                 data_format=None):
        if not data_format:
            data_format = ('channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

        self.data_format = data_format
        self.filter_sizes = filter_sizes
        self.num_convs = num_convs
        self.num_lmark = num_lmark
        self.kernel_size = kernel_size
        self.img_size = img_size

        self.__pixels__ = tf.constant([(x, y) for y in range(self.img_size) for x in range(self.img_size)],
                                      dtype=tf.float32, shape=[1, self.img_size, self.img_size, 2])
        # self.__pixels__ = tf.tile(self.__pixels__,[num_lmark,1,1,1])

        # self.IMGSIZE = 112
        # self.N_LANDMARK = 68

        self.Pixels = tf.constant(
            np.array([(x, y) for x in range(self.img_size) for y in range(self.img_size)], dtype=np.float32), \
            shape=[self.img_size, self.img_size, 2])

    def AffineTransformLayer(self, image, r, t):
        """
        Image: [N, IMGSIZE, IMGSIZE, 2]
        Param: [N, 6]
        return: [N, IMGSIZE, IMGSIZE, 2]
        """

        IMGSIZE = self.img_size
        Pixels = self.Pixels

        # A = tf.reshape(affine_param[:, 0:4], (-1, 2, 2))
        # T = tf.reshape(affine_param[:, 4:6], (-1, 1, 2))

        A = r
        T = t

        A = tf.matrix_inverse(A)
        T = tf.matmul(-T, A)

        T = tf.reverse(T, (-1,))
        A = tf.matrix_transpose(A)

        def affine_transform(I, A, T):
            I = tf.reshape(I, [IMGSIZE, IMGSIZE])

            SrcPixels = tf.matmul(tf.reshape(Pixels, [IMGSIZE * IMGSIZE, 2]), A) + T
            SrcPixels = tf.clip_by_value(SrcPixels, 0, IMGSIZE - 2)

            outPixelsMinMin = tf.to_float(tf.to_int32(SrcPixels))
            dxdy = SrcPixels - outPixelsMinMin
            dx = dxdy[:, 0]
            dy = dxdy[:, 1]

            outPixelsMinMin = tf.reshape(tf.to_int32(outPixelsMinMin), [IMGSIZE * IMGSIZE, 2])
            outPixelsMaxMin = tf.reshape(outPixelsMinMin + [1, 0], [IMGSIZE * IMGSIZE, 2])
            outPixelsMinMax = tf.reshape(outPixelsMinMin + [0, 1], [IMGSIZE * IMGSIZE, 2])
            outPixelsMaxMax = tf.reshape(outPixelsMinMin + [1, 1], [IMGSIZE * IMGSIZE, 2])

            OutImage = (1 - dx) * (1 - dy) * tf.gather_nd(I, outPixelsMinMin) + dx * (1 - dy) * tf.gather_nd(I,
                                                                                                             outPixelsMaxMin) \
                       + (1 - dx) * dy * tf.gather_nd(I, outPixelsMinMax) + dx * dy * tf.gather_nd(I, outPixelsMaxMax)

            return tf.reshape(OutImage, [IMGSIZE, IMGSIZE, 1])

        return tf.map_fn(lambda args: affine_transform(args[0], args[1], args[2]), (image, A, T), dtype=tf.float32)

    def TransformParamsLayer(self, src_shapes, dst_shape):
        """
        SrcShapes: [N, (N_LANDMARK x 2)]
        DstShape: [N_LANDMARK x 2,]
        return: [N, 6]
        """

        # import pdb; pdb.set_trace()
        def bestFit(src, dst):
            # import pdb; pdb.set_trace()
            source = tf.reshape(src, (-1, 2))
            destination = tf.reshape(dst, (-1, 2))

            destMean = tf.reduce_mean(destination, axis=0)
            srcMean = tf.reduce_mean(source, axis=0)

            srcCenter = source - srcMean
            dstCenter = destination - destMean

            srcVec = tf.reshape(srcCenter, (-1,))
            destVec = tf.reshape(dstCenter, (-1,))
            norm = (tf.norm(srcVec) ** 2)

            a = tf.tensordot(srcVec, destVec, 1) / norm
            b = 0

            srcX = tf.reshape(srcVec, (-1, 2))[:, 0]
            srcY = tf.reshape(srcVec, (-1, 2))[:, 1]
            destX = tf.reshape(destVec, (-1, 2))[:, 0]
            destY = tf.reshape(destVec, (-1, 2))[:, 1]

            b = tf.reduce_sum(tf.multiply(srcX, destY) - tf.multiply(srcY, destX))
            b = b / norm

            A = tf.reshape(tf.stack([a, b, -b, a]), (2, 2))
            srcMean = tf.tensordot(srcMean, A, 1)

            return tf.concat((tf.reshape(A, (-1,)), destMean - srcMean), 0)

        return tf.map_fn(lambda s: bestFit(s, dst_shape), src_shapes)

    def __calc_affine_params(self, from_shape, to_shape):
        from_shape = tf.reshape(from_shape, [-1, self.num_lmark, 2])
        to_shape = tf.reshape(to_shape, [-1, self.num_lmark, 2])

        from_mean = tf.reduce_mean(from_shape, axis=1, keepdims=True)
        to_mean = tf.reduce_mean(to_shape, axis=1, keepdims=True)

        from_centralized = from_shape - from_mean
        to_centralized = to_shape - to_mean

        dot_result = tf.reduce_sum(tf.multiply(from_centralized, to_centralized),
                                   axis=[1, 2])
        norm_pow_2 = tf.pow(tf.norm(from_centralized, axis=[1, 2]), 2)

        a = dot_result / norm_pow_2
        b = tf.reduce_sum(
            tf.multiply(from_centralized[:, :, 0], to_centralized[:, :, 1]) - tf.multiply(from_centralized[:, :, 1],
                                                                                          to_centralized[:, :, 0]),
            1) / norm_pow_2

        r = tf.reshape(tf.stack([a, b, -b, a], axis=1), [-1, 2, 2])
        t = to_mean - tf.matmul(from_mean, r)
        return r, t

    def __affine_image(self, imgs, r, t):
        # The Tensor [imgs].format is [NHWC]
        r = tf.matrix_inverse(r)
        r = tf.matrix_transpose(r)

        rm = tf.reshape(tf.pad(r, [[0, 0], [0, 0], [0, 1]], mode='CONSTANT'), [-1, 6])
        rm = tf.pad(rm, [[0, 0], [0, 2]], mode='CONSTANT')

        tm = tf.contrib.image.translations_to_projective_transforms(tf.reshape(t, [-1, 2]))
        rtm = tf.contrib.image.compose_transforms(rm, tm)

        return tf.contrib.image.transform(imgs, rtm, "BILINEAR")

    def __affine_shape(self, shapes, r, t, isinv=False):
        if isinv:
            r = tf.matrix_inverse(r)
            t = tf.matmul(-t, r)
        shapes = tf.matmul(shapes, r) + t
        return shapes

    def __gen_heatmap(self, shapes):
        shapes = shapes[:, :, tf.newaxis, tf.newaxis, :]
        value = self.__pixels__ - shapes
        value = tf.norm(value, axis=-1)
        value = 1.0 / (tf.reduce_min(value, axis=1) + 1.0)
        value = tf.expand_dims(value, axis=-1)
        return value

    def __call__(self,
                 inputs_imgs,
                 s1_training,
                 s2_training,
                 mean_shape,
                 imgs_mean,
                 imgs_std):
        rd = {}
        inputs_imgs = tf.reshape(inputs_imgs, [-1, self.img_size, self.img_size, 1])
        tf.summary.image('image', inputs_imgs, max_outputs=6)

        rd['img'] = inputs_imgs

        mean_shape = tf.reshape(mean_shape, [self.num_lmark, 2]) if mean_shape is not None else tf.zeros(
            [self.num_lmark, 2], tf.float32)
        imgs_mean = tf.reshape(imgs_mean, [self.img_size, self.img_size, 1]) if imgs_mean is not None else tf.zeros(
            [self.img_size, self.img_size, 1], tf.float32)
        imgs_std = tf.reshape(imgs_std, [self.img_size, self.img_size, 1]) if imgs_std is not None else tf.ones(
            [self.img_size, self.img_size, 1], tf.float32)

        imgs_mean_tensor = tf.get_variable('imgs_mean', trainable=False, initializer=imgs_mean)
        imgs_std_tensor = tf.get_variable('imgs_std', trainable=False, initializer=imgs_std)
        shape_mean_tensor = tf.get_variable('shape_mean', trainable=False, initializer=mean_shape)

        inputs_imgs = (inputs_imgs - imgs_mean_tensor) / imgs_std_tensor
        # Convert the inputs from channels_last (NHWC) to channels_first
        # (NCHW).
        # This provides a large performance boost on GPU.  See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        with tf.variable_scope('s1'):
            inputs = inputs_imgs

            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            for i, num_filter in enumerate(self.filter_sizes):
                inputs = vgg_block(inputs=inputs, filters=num_filter, num_convs=self.num_convs,
                                   training=s1_training, kernel_size=self.kernel_size, maxpool=True,
                                   data_format=self.data_format)

            inputs = tf.contrib.layers.flatten(inputs)
            inputs = tf.layers.dropout(inputs, 0.5, training=s1_training)

            s1_fc1 = tf.layers.dense(inputs, 256, activation=tf.nn.relu,
                                     kernel_initializer=tf.glorot_uniform_initializer())
            s1_fc1 = batch_norm(s1_fc1, s1_training, data_format=self.data_format)

            s1_fc2 = tf.layers.dense(s1_fc1, self.num_lmark * 2, activation=None)
            rd['s1_ret'] = tf.identity(tf.reshape(s1_fc2, [-1, self.num_lmark, 2]) + shape_mean_tensor,
                                       name='output_landmark')

        with tf.variable_scope('s2'):
            r, t = self.__calc_affine_params(rd['s1_ret'], shape_mean_tensor)
            # inputs = self.__affine_image(inputs_imgs, r, t)

            # affine_param = self.TransformParamsLayer(rd['s1_ret'], shape_mean_tensor)
            # affine_param = [r, t]
            inputs = self.AffineTransformLayer(inputs_imgs, r, t)
            inputs = tf.reshape(inputs, (-1, self.img_size, self.img_size, 1))	

            s2_lmark = self.__affine_shape(rd['s1_ret'], r, t)
            s2_heatmap = self.__gen_heatmap(s2_lmark)
            s2_feature = tf.layers.dense(s1_fc1, (self.img_size // 2) ** 2, activation=tf.nn.relu,
                                         kernel_initializer=tf.glorot_uniform_initializer())

            s2_feature = tf.reshape(s2_feature, [-1, self.img_size // 2, self.img_size // 2, 1])
            s2_feature_upscale = tf.image.resize_images(s2_feature, [self.img_size, self.img_size])

            tf.summary.image('heatmap', s2_heatmap, max_outputs=6)
            tf.summary.image('feature', s2_feature, max_outputs=6)
            tf.summary.image('image', inputs, max_outputs=6)

            if self.data_format == 'channels_first':
                inputs = tf.transpose(inputs, [0, 3, 1, 2])
                s2_heatmap = tf.transpose(s2_heatmap, [0, 3, 1, 2])
                s2_feature_upscale = tf.transpose(s2_feature_upscale, [0, 3, 1, 2])

            inputs = tf.concat([inputs, s2_heatmap, s2_feature_upscale],
                               axis=1 if self.data_format == 'channels_first' else 3)
            inputs = batch_norm(inputs, s2_training, self.data_format)

            for i, num_filter in enumerate(self.filter_sizes):
                inputs = vgg_block(inputs=inputs, filters=num_filter, num_convs=self.num_convs,
                                   training=s2_training, kernel_size=self.kernel_size, maxpool=True,
                                   data_format=self.data_format)

            inputs = tf.contrib.layers.flatten(inputs)
            inputs = tf.layers.dropout(inputs, 0.5, training=s2_training)

            s2_fc1 = tf.layers.dense(inputs, 256, activation=tf.nn.relu,
                                     kernel_initializer=tf.glorot_uniform_initializer())
            s2_fc1 = batch_norm(s2_fc1, s2_training, data_format=self.data_format)

            s2_fc2 = tf.layers.dense(s2_fc1, self.num_lmark * 2, activation=None)
            s2_fc2 = tf.reshape(s2_fc2, [-1, self.num_lmark, 2]) + s2_lmark
            rd['s2_ret'] = tf.identity(self.__affine_shape(s2_fc2, r, t, isinv=True), name='output_landmark')

        return rd
