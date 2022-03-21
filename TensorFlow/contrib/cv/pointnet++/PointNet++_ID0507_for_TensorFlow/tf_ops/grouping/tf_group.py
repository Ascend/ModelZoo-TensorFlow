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
import numpy as np
import tensorflow as tf

def query_ball_point(radius, nsample, xyz, new_xyz):
    '''

    Args:
        radius: float32 local region radius
        nsample: max sample number in local region int32
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]

    Returns:
        group_idx: grouped points index, [B, S, nsample]

    '''
    B = int(xyz.get_shape()[0])
    N = int(xyz.get_shape()[1])
    C = int(xyz.get_shape()[2])
    S = int(new_xyz.get_shape()[1])
    K = nsample
    group_idx = np.arange(N)
    group_idx = group_idx.reshape((1, 1, N))
    group_idx = group_idx.repeat(S, axis=1)
    group_idx = group_idx.repeat(B, axis=0)
    group_idx = tf.convert_to_tensor(group_idx, dtype=tf.int32)
    sqrdists = _square_distance(new_xyz, xyz)
    N_tensor = tf.constant(N, dtype=tf.int32, shape=[B, S, N])
    group_idx = tf.where(tf.greater(sqrdists, (radius ** 2)), N_tensor, group_idx)
    group_idx = tf.sort(group_idx, direction='ASCENDING')[:, :, :nsample]
    group_first = tf.repeat(tf.reshape(group_idx[:, :, 0], [B,S,1]), nsample, axis=2)
    group_idx = tf.where(tf.equal(group_idx, N), group_first, group_idx)
    return group_idx

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

def _square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * tf.matmul(src, tf.transpose(dst, perm=(0, 2, 1)))
    dist += tf.reshape(tf.reduce_sum(src ** 2, -1), [B, N, 1])
    dist += tf.reshape(tf.reduce_sum(dst ** 2, -1), [B, 1, M])
    return dist

if __name__ == '__main__':
    import numpy as np
    np.random.seed(100)
    points = np.random.rand(3, 10, 3).astype('float32')
    points = tf.constant(points)
    new_xyz = [
        [[0.84477615, 0.00471886, 0.12156912],
         [0.9786238, 0.8116832, 0.17194101]],
        [[0.5446849, 0.76911515, 0.25069523],
         [0.3547956, 0.3401902, 0.17808099]],
        [[0.37625244, 0.5928054, 0.6299419],
         [0.08846017, 0.5280352, 0.99215806]]
    ]
    new_xyz = tf.constant(new_xyz, dtype=tf.float32)
    group_idx = query_ball_point(0.5, 4, points, new_xyz)
    resample_points = group_point(points, group_idx)
    with tf.Session(config=npu_config_proto()) as sess:
        print(resample_points.eval())
