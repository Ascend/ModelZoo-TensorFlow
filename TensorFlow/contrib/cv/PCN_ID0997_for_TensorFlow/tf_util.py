'''
MIT License

Copyright (c) 2018 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
from npu_bridge.npu_init import *

import tensorflow as tf
from pc_distance import tf_nndistance, tf_approxmatch


def mlp(features, layer_dims, bn=None, bn_params=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv1d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    outputs = tf.contrib.layers.conv1d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    return outputs


def point_maxpool(inputs, npts, keepdims=False):
    inputs = [inputs[:,3000*i:3000*(i+1),:] for i in range(npts.shape[0])]
    #print(tf.split(inputs, npts, axis=1))
    outputs = [tf.reduce_max(f, axis=1, keepdims=keepdims)
        for f in inputs]#tf.split(inputs, npts, axis=1)]
    return tf.concat(outputs, axis=0)


def point_unpool(inputs, npts):
    inputs = [tf.expand_dims(inputs[i,...],0) for i in range(inputs.shape[0])]
    #inputs2 = tf.split(inputs, inputs.shape[0], axis=0)
    #print(inputs2)
    #print(inputs)
    outputs = [tf.tile(f, [1, 3000, 1]) for i,f in enumerate(inputs)]
    return tf.concat(outputs, axis=1)

def chamfer(pcd1, pcd2):
    print('chamfer:',pcd1,pcd2)
    dist1, _, dist2, _ = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1))
    dist2 = tf.reduce_mean(tf.sqrt(dist2))

    """
      this is an official implementation by tensorflow-graphics/google
      project url:https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/nn/loss/chamfer_distance.py 
      Computes the Chamfer distance for the given two point sets.
      Note:
        This is a symmetric version of the Chamfer distance, calculated as the sum
        of the average minimum distance from point_set_a to point_set_b and vice
        versa.
        The average minimum distance from one point set to another is calculated as
        the average of the distances between the points in the first set and their
        closest point in the second set, and is thus not symmetrical.
      Note:
        This function returns the exact Chamfer distance and not an approximation.
      Note:
        In the following, A1 to An are optional batch dimensions, which must be
        broadcast compatible.
      Args:
        point_set_a: A tensor of shape `[A1, ..., An, N, D]`, where the last axis
          represents points in a D dimensional space.
        point_set_b: A tensor of shape `[A1, ..., An, M, D]`, where the last axis
          represents points in a D dimensional space.
        name: A name for this op. Defaults to "chamfer_distance_evaluate".
      Returns:
        A tensor of shape `[A1, ..., An]` storing the chamfer distance between the
        two point sets.
      Raises:
        ValueError: if the shape of `point_set_a`, `point_set_b` is not supported.
    point_set_a = tf.convert_to_tensor(value=pcd1)
    point_set_b = tf.convert_to_tensor(value=pcd2)

    shape.compare_batch_dimensions(
        tensors=(point_set_a, point_set_b),
        tensor_names=("point_set_a", "point_set_b"),
        last_axes=-3,
        broadcast_compatible=True)
    # Verify that the last axis of the tensors has the same dimension.
    dimension = point_set_a.shape.as_list()[-1]
    shape.check_static(
        tensor=point_set_b,
        tensor_name="point_set_b",
        has_dim_equals=(-1, dimension))

        # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of
        # dimension D).
    difference = (
            tf.expand_dims(point_set_a, axis=-2) -
            tf.expand_dims(point_set_b, axis=-3))
    # Calculate the square distances between each two points: |ai - bj|^2.
    square_distances = tf.einsum("...i,...i->...", difference, difference)

    minimum_square_distance_a_to_b = tf.reduce_min(
        input_tensor=square_distances, axis=-1)
    minimum_square_distance_b_to_a = tf.reduce_min(
        input_tensor=square_distances, axis=-2)

    return (tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
            tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))
    """
    return (dist1 + dist2) / 2
    
#def earth_mover(pcd1,pcd2):
#
#    return

def earth_mover(pcd1, pcd2):
    assert pcd1.shape[1] == pcd2.shape[1]
    num_points = tf.cast(pcd1.shape[1], tf.float32)
    match = tf_approxmatch.approx_match(pcd1, pcd2)
    print(match)
    cost = tf_approxmatch.match_cost(pcd1, pcd2, match)
    print('costshape:',cost)
    return tf.reduce_mean(cost / num_points)


def add_train_summary(name, value):
    tf.compat.v1.summary.scalar(name, value, collections=['train_summary'])


def add_valid_summary(name, value):
    avg, update = tf.compat.v1.metrics.mean(value)
    tf.compat.v1.summary.scalar(name, avg, collections=['valid_summary'])
    return update

