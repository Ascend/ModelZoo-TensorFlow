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
import numpy as np
from utils.ThinPlateSplineB import ThinPlateSpline3 as TPS


def image_warping(img):
    t_ = np.array([  # target position
        [-1., -1.],
        [1., -1.],
        [-1., 1.],
        [1., 1.],
        [0., 1.],
        [0., -1.],
        [1., 0.],
        [-1., 0.],
        [0., 0.],

    ])
    grid = tf.constant(t_.reshape([1, 9, 2]), dtype=tf.float32)

    CROP_SIZE = img.get_shape()[1]

    deformation = tf.random_uniform([1, 1, 2], minval=-.5, maxval=.5, dtype=tf.float32)

    grid_deformed = grid[:, 0:8, :]
    grid_deformed = tf.concat([grid_deformed, deformation], axis=1)
    input_images_expanded = tf.reshape(img, [1, CROP_SIZE, CROP_SIZE, 3, 1])
    print(input_images_expanded.get_shape())

    t_img = TPS(input_images_expanded, grid_deformed, grid, [CROP_SIZE, CROP_SIZE, 3])
    t_img = tf.reshape(t_img, tf.shape(img))

    print(deformation.get_shape())

    return t_img, deformation

def image_warping2(img, w):
    t_ = np.array([  # target position
        [-1., -1.],
        [1., -1.],
        [-1., 1.],
        [1., 1.],
    ])
    grid = tf.constant(t_.reshape([1, 4, 2]), dtype=tf.float32)
    CROP_SIZE = img.get_shape()[1]
    rotation = tf.random_uniform([1, 1], minval=-0.5, maxval=0.5, dtype=tf.float32)
    x_translation = tf.random_normal([1, 1], mean=0., stddev=0.0 + w, dtype=tf.float32)
    y_translation = tf.random_normal([1, 1], mean=0., stddev=0.0 + w, dtype=tf.float32)
    x_scale = tf.random_uniform([1, 1], minval=0.8-w, maxval=1.1+w, dtype=tf.float32)
    y_scale = x_scale + tf.random_normal([1, 1], mean=0.0, stddev=0.1, dtype=tf.float32)#tf.random_uniform([1, 1], minval=0.6, maxval=1.2, dtype=tf.float32)
    a1 = tf.concat([x_translation, x_scale*tf.cos(rotation), -1.*y_scale*tf.sin(rotation)], axis=1)
    a2 = tf.concat([y_translation, x_scale*tf.sin(rotation), y_scale*tf.cos(rotation)], axis=1)
    A = tf.concat([a1, a2], axis=0)
    zero = tf.zeros([2, 4], tf.float32)
    T = tf.concat([A, zero], axis=1)
    T = tf.expand_dims(T, axis=0)
    print('img>>>>>>>>>>>>',img,"CROP_SIZE>>>>>>>>>",CROP_SIZE)
    input_images_expanded = tf.reshape(img, [1, CROP_SIZE, CROP_SIZE, 3, 1])

    t_img = TPS(input_images_expanded, grid, T, [CROP_SIZE, CROP_SIZE, 3])

    t_img = tf.reshape(t_img, tf.shape(img))

    return t_img, T


def feature_warping(feature, deformation):
    t_ = np.array([  # target position
        [-1., -1.],
        [1., -1.],
        [-1., 1.],
        [1., 1.],
        [0., 1.],
        [0., -1.],
        [1., 0.],
        [-1., 0.],
        [0., 0.],

    ])

    CROP_SIZE = feature.get_shape()[1]
    Batch_SIZE = feature.get_shape()[0]
    DEPTH = feature.get_shape()[3]

    grid = tf.constant(t_.reshape([1, 9, 2]), dtype=tf.float32)
    grid = tf.tile(grid, [Batch_SIZE, 1, 1])

    # deformation = tf.random_uniform([1, 1, 2], minval=-.5, maxval=.5, dtype=tf.float32)

    grid_deformed = grid[:, 0:8, :]

    print(grid_deformed.get_shape())
    print(deformation.get_shape())

    deformation = tf.reshape(deformation, [Batch_SIZE, 1, 2])

    grid_deformed = tf.concat([grid_deformed, deformation], axis=1)

    input_images_expanded = tf.reshape(feature, [Batch_SIZE, CROP_SIZE, CROP_SIZE, DEPTH, 1])
    print(input_images_expanded.get_shape())

    t_img = TPS(input_images_expanded, grid_deformed, grid, [CROP_SIZE, CROP_SIZE, DEPTH])
    print(t_img.get_shape())
    t_img = tf.reshape(t_img, tf.shape(feature))
    print(t_img.get_shape())

    return t_img

def feature_warping2(feature, deformation, padding=0):
    t_ = np.array([  # target position
        [-1., -1.],
        [1., -1.],
        [-1., 1.],
        [1., 1.],
    ])
    feature = tf.pad(feature, [[0, 0], [padding, padding], [padding, padding], [0, 0]], "CONSTANT")
    CROP_SIZE = feature.get_shape()[1]
    Batch_SIZE = feature.get_shape()[0]
    DEPTH = feature.get_shape()[3]

    grid = tf.constant(t_.reshape([1, 4, 2]), dtype=tf.float32)
    grid = tf.tile(grid, [Batch_SIZE, 1, 1])

    input_images_expanded = tf.reshape(feature, [Batch_SIZE, CROP_SIZE, CROP_SIZE, DEPTH, 1])
    t_img = TPS(input_images_expanded, grid, deformation, [CROP_SIZE, CROP_SIZE, DEPTH])
    t_img = tf.reshape(t_img, tf.shape(feature))
    t_img = tf.image.crop_to_bounding_box(t_img, padding, padding, CROP_SIZE-2*padding, CROP_SIZE-2*padding)
    return t_img