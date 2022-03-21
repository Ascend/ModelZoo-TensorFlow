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


from npu_bridge.npu_init import *
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os

def three_nn(xyz1, xyz2):
    '''
    Input:
        xyz1: (b,n,3) float32 array, unknown points
        xyz2: (b,m,3) float32 array, known points
    Output:
        dist: (b,n,3) float32 array, distances to known points
        idx: (b,n,3) int32 array, indices to known points
    '''
    dist = _square_distance(xyz1, xyz2)
    idx = tf.argsort(dist, direction='ASCENDING')
    dist = tf.sort(dist, direction='ASCENDING')
    idx = idx[:, :, :3]
    dist = dist[:, :, :3]
    return dist, idx
ops.NoGradient('ThreeNN')


def three_interpolate(points, idx, weight):
    '''
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,3) int32 array, indices to known points
        weight: (b,n,3) float32 array, weights on known points
    Output:
        out: (b,n,c) float32 array, interpolated point values
    '''
    B = int(idx.get_shape()[0])
    N = int(idx.get_shape()[1])
    weight = tf.reshape(weight, [B, N, 3, 1])
    interpolated_point = group_point(points, idx) * weight
    interpolated_point = tf.reduce_sum(interpolated_point, axis=2)
    return interpolated_point


# @tf.RegisterGradient('ThreeInterpolate')
# def _three_interpolate_grad(op, grad_out):
#     points = op.inputs[0]
#     idx = op.inputs[1]
#     weight = op.inputs[2]
#     return [interpolate_module.three_interpolate_grad(points, idx, weight, grad_out), None, None]None

def _square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * tf.matmul(src, tf.transpose(dst, perm=(0, 2, 1)))
    dist += tf.reshape(tf.reduce_sum(src ** 2, -1), [B, N, 1])
    dist += tf.reshape(tf.reduce_sum(dst ** 2, -1), [B, 1, M])
    return dist

def group_point(points, idx):
    '''

    Args:
        points: [batch_size, ndataset, channel) float32
        idx: (batch_size, npoint, nsample) int32

    Returns:
        out: (batch_size, npoint, nsample, channel) float32
    '''
    bz = int(points.get_shape()[0])
    nd = int(points.get_shape()[1])
    channel = int(points.get_shape()[2])
    npoint = int(idx.get_shape()[1])
    sample_points = None
    if bz > 0:
        sample_points = tf.gather(points[0], idx[0], axis=0)
        sample_points = tf.expand_dims(sample_points, axis=0)

    for i in range(1, bz):
        sample_point = tf.gather(points[i], idx[i], axis=0)
        sample_point = tf.expand_dims(sample_point, axis=0)
        sample_points = tf.concat([sample_points, sample_point], 0)
    return sample_points


if __name__ == '__main__':
    import numpy as np

    np.random.seed(100)
    xyz1 = np.random.rand(3, 10, 3).astype('float32')
    xyz1 = tf.constant(xyz1)
    xyz2 = np.random.rand(3, 5, 3).astype('float32')
    xyz2 = tf.constant(xyz2)
    dist, idx = three_nn(xyz1, xyz2)
    res = three_interpolate(xyz2, idx, xyz1)
    with tf.Session(config=npu_config_proto()) as sess:
        print(res.eval())
