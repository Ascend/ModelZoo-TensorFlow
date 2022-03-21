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


''' Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017.
'''
from npu_bridge.npu_init import *
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np

# ops.NoGradient('ProbSample')


# TF1.0 API requires set shape in C++
# @tf.RegisterShape('ProbSample')
# def _prob_sample_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(2)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
def gather_point(inp, idx):
    '''
input:
    batch_size * ndataset * 3   float32
    batch_size * npoints        int32
returns:
    batch_size * npoints * 3    float32
    '''
    bz, nd, channel = inp.shape
    sample_points = None
    if bz > 0:
        sample_points = tf.gather(inp[0], idx[0], axis=0)
        sample_points = tf.expand_dims(sample_points, axis=0)

    for i in range(1, bz):
        sample_point = tf.gather(inp[i], idx[i], axis=0)
        sample_point = tf.expand_dims(sample_point, axis=0)
        sample_points = tf.concat([sample_points, sample_point], 0)
    return sample_points


# @tf.RegisterShape('GatherPoint')
# def _gather_point_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(3)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape1.dims[0],shape2.dims[1],shape1.dims[2]])]
# @tf.RegisterGradient('GatherPoint')
# def _gather_point_grad(op, out_g):
#     inp = op.inputs[0]
#     idx = op.inputs[1]
#     return [sampling_module.gather_point_grad(inp, idx, out_g), None]


def farthest_point_sample(npoint, inp):
    '''
input:
    int32
    batch_size * ndataset * 3   float32
returns:
    batch_size * npoint         int32
    '''
    B = int(inp.get_shape()[0])
    N = int(inp.get_shape()[1])
    C = int(inp.get_shape()[2])
    S = npoint
    centroids = tf.zeros([B, S], dtype=tf.int64)
    distance = tf.ones([B, N]) * 1e3
    farthest = np.random.randint(0, N, size=B)
    farthest = tf.convert_to_tensor(farthest, dtype=tf.int64)
    for i in range(S):
        centroids = modify_tensor(centroids, i, farthest)
        centroid = slice_tensor(inp, farthest)
        dist = tf.reduce_sum((inp - centroid) ** 2, 2)
        distance = tf.where(tf.greater(distance, dist), dist, distance)
        farthest = tf.arg_max(distance, 1)
    return centroids

ops.NoGradient('FarthestPointSample')

def modify_tensor(inp, idx, tar):
    B, S = inp.shape
    tar1 = tf.reshape(tar, [B, 1])
    part1 = inp[:, :idx]
    part2 = inp[:, idx+1:]
    res = tf.concat([part1, tar1, part2], axis=1)
    return res

def slice_tensor(inp, farthest):
    B, S, C = inp.shape
    if B > 0:
        parts = inp[0, farthest[0]]
        parts = tf.expand_dims(parts, axis=0)
        parts = tf.expand_dims(parts, axis=0)
    for i in range(1, B):
        idx = farthest[i]
        part = inp[i, idx]
        part = tf.expand_dims(part, axis=0)
        part = tf.expand_dims(part, axis=0)
        parts = tf.concat([parts, part], axis=0)
    return parts

# def set_value1(matrix, x, y, val):
#     # 提取出要更新的行
#     row = tf.gather(matrix, x)
#     # 构造这行的新数据
#     new_row = tf.concat([row[:y], [val], row[y+1:]], axis=0)
#     # 使用 tf.scatter_update 方法进正行替换
#     matrix.assign(tf.scatter_update(matrix, x, new_row))
#
# def set_value(matrix, x, y, val):
#     # 得到张量的宽和高，即第一维和第二维的Size
#     w = int(matrix.get_shape()[0])
#     h = int(matrix.get_shape()[1])
#     # 构造一个只有目标位置有值的稀疏矩阵，其值为目标值于原始值的差
#     val_diff = val - matrix[x][y]
#     diff_matrix = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[[x, y]], values=[val_diff], dense_shape=[w, h]))
#     # 用 Variable.assign_add 将两个矩阵相加
#     return matrix + diff_matrix



if __name__ == '__main__':
    import numpy as np

    np.random.seed(100)
    points = np.random.rand(3, 10, 3).astype('float32')
    points = tf.constant(points)
    res = farthest_point_sample(4, points)

    sample_points = gather_point(points, res)

