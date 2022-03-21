#
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
import numpy as np
import tensorflow as tf

from .layers import conv2d_bn_relu
from .resnet import resnet18


class CenterNet():
    """CenterNet model
    """
    BACKBONE_CONFIG = {'resnet-18': {'model': resnet18}}

    def __init__(self,
                 is_training,
                 num_classes,
                 backbone='resnet-18',
                 deform_conv=tf.layers.conv2d):
        """CenterNet

        Args:
            is_training (bool):
            num_classes ([int]):
            backbone (str, optional):Defaults to 'resnet-18'.
            deform_conv (optional): conv used. Defaults to tf.layers.conv2d.
        """
        assert backbone in self.BACKBONE_CONFIG, 'backbone does not support'
        self.backbone = backbone
        self.is_training = is_training
        self.num_classes = num_classes
        self.deform_conv = deform_conv

    def _make_upsampling(self, backbone_stages):
        """make upsampling layers

        Args:
            backbone_stages:

        Returns:
            upsampling layers output
        """
        if self.backbone == 'resnet-18':
            x = backbone_stages[-1]
            for filters in [256, 128, 64]:
                x = conv2d_bn_relu(x,
                                   filters, (3, 3),
                                   1,
                                   'same',
                                   self.deform_conv,
                                   is_training=self.is_training)
                x = conv2d_bn_relu(x,
                                   filters, (4, 4),
                                   2,
                                   'same',
                                   tf.layers.conv2d_transpose,
                                   is_training=self.is_training)
            return x

    def call(self, inputs):
        """call

        Args:
            inputs:

        Returns:
            heatmap:
            size:
            offset:
        """
        with tf.name_scope('backbone'):
            backbone_stages = self.BACKBONE_CONFIG[self.backbone]['model'](
                inputs, is_training=self.is_training)
        with tf.name_scope('upsampling'):
            features = self._make_upsampling(backbone_stages)
        with tf.name_scope('detector'):
            heatmap = tf.layers.conv2d(features,
                                       64, (3, 3),
                                       1,
                                       'same',
                                       activation=tf.nn.relu)
            heatmap = tf.layers.conv2d(heatmap, self.num_classes, (1, 1))
            size = tf.layers.conv2d(features,
                                    64, (3, 3),
                                    1,
                                    'same',
                                    activation=tf.nn.relu)
            size = tf.layers.conv2d(size, 2, (1, 1))
            offset = tf.layers.conv2d(features,
                                      64, (3, 3),
                                      1,
                                      'same',
                                      activation=tf.nn.relu)
            offset = tf.layers.conv2d(offset, 2, (1, 1))
        return heatmap, size, offset


def _gaussian_radius(height, width, min_overlap=0.7):
    """gaussian radius

    Args:
        height:
        width:
        min_overlap (float, optional): Defaults to 0.7.

    Returns:
        gaussian radius
    """
    # Modified from https://github.com/Stick-To/CenterNet-tensorflow/blob/master/CenterNet.py _gaussian_radius
    # BugFixed according to https://github.com/princeton-vl/CornerNet/blob/e5c39a31a8abef5841976c8eab18da86d6ee5f9a/sample/util.py gaussian_radius
    a1 = 1.
    b1 = (height + width)
    c1 = width * height * (1. - min_overlap) / (1. + min_overlap)
    sq1 = tf.sqrt(b1**2. - 4. * a1 * c1)
    r1 = (b1 + sq1) / (2. * a1)

    a2 = 4.
    b2 = 2. * (height + width)
    c2 = (1. - min_overlap) * width * height
    sq2 = tf.sqrt(b2**2. - 4. * a2 * c2)
    r2 = (b2 + sq2) / (2. * a2)

    a3 = 4. * min_overlap
    b3 = -2. * min_overlap * (height + width)
    c3 = (min_overlap - 1.) * width * height
    sq3 = tf.sqrt(b3**2. - 4. * a3 * c3)
    r3 = (b3 + sq3) / (2. * a3)
    return tf.reduce_min([r1, r2, r3])


def _heatmap_loss(heatmap, gbbox_yx, gbbox_y, gbbox_x, gbbox_h, gbbox_w,
                  classid, meshgrid_y, meshgrid_x, pshape, num_classes):
    """heatmap loss

    Args:
        heatmap
        gbbox_yx
        gbbox_y
        gbbox_x
        gbbox_h
        gbbox_w
        classid
        meshgrid_y
        meshgrid_x
        pshape
        num_classes

    Returns:
        heatmap_loss
    """
    sigma = _gaussian_radius(gbbox_h, gbbox_w, 0.7)
    gbbox_y = tf.reshape(gbbox_y, [-1, 1, 1])
    gbbox_x = tf.reshape(gbbox_x, [-1, 1, 1])
    sigma = tf.reshape(sigma, [-1, 1, 1])

    num_g = tf.shape(gbbox_y)[0]
    meshgrid_y = tf.expand_dims(meshgrid_y, 0)
    meshgrid_y = tf.tile(meshgrid_y, [num_g, 1, 1])
    meshgrid_x = tf.expand_dims(meshgrid_x, 0)
    meshgrid_x = tf.tile(meshgrid_x, [num_g, 1, 1])

    keyp_penalty_reduce = tf.exp(-((gbbox_y - meshgrid_y)**2 +
                                   (gbbox_x - meshgrid_x)**2) / (2 * sigma**2))
    zero_like_keyp = tf.expand_dims(tf.zeros(pshape, dtype=tf.float32),
                                    axis=-1)
    reduction = []
    gt_heatmap = []
    for i in range(num_classes):
        exist_i = tf.equal(classid, i)
        reduce_i = tf.boolean_mask(keyp_penalty_reduce, exist_i, axis=0)
        reduce_i = tf.cond(
            tf.equal(tf.shape(reduce_i)[0], 0), lambda: zero_like_keyp,
            lambda: tf.expand_dims(tf.reduce_max(reduce_i, axis=0), axis=-1))
        reduction.append(reduce_i)

        gbbox_yx_i = tf.boolean_mask(gbbox_yx, exist_i)
        gt_heatmap_i = tf.cond(
            tf.equal(tf.shape(gbbox_yx_i)[0], 0), lambda: zero_like_keyp,
            lambda: tf.expand_dims(tf.sparse.to_dense(tf.sparse.SparseTensor(
                gbbox_yx_i,
                tf.ones_like(gbbox_yx_i[..., 0], tf.float32),
                dense_shape=pshape),
                                                      validate_indices=False),
                                   axis=-1))
        gt_heatmap.append(gt_heatmap_i)
    reduction = tf.concat(reduction, axis=-1)
    gt_heatmap = tf.concat(gt_heatmap, axis=-1)
    heatmap_pos_loss = -tf.pow(1. - tf.sigmoid(heatmap),
                               2.) * tf.log_sigmoid(heatmap) * gt_heatmap
    heatmap_neg_loss = -tf.pow(1. - reduction, 4) * tf.pow(
        tf.sigmoid(heatmap),
        2.) * (-heatmap + tf.log_sigmoid(heatmap)) * (1. - gt_heatmap)
    num_g = tf.maximum(num_g, tf.ones_like(num_g, dtype=tf.int32))
    heatmap_loss = tf.reduce_sum(heatmap_pos_loss) / tf.cast(
        num_g, tf.float32) + tf.reduce_sum(heatmap_neg_loss) / tf.cast(
            num_g, tf.float32)
    return heatmap_loss


def _cal_single_loss(heatmap, size, offset, ground_truth, stride, pshape,
                     num_classes, meshgrid_y, meshgrid_x):
    """calculate single pic loss

    Args:
        heatmap
        size
        offset
        ground_truth
        stride
        pshape
        num_classes
        meshgrid_y
        meshgrid_x

    Returns:
        heatmap_loss
        size_loss
        offset_loss
    """
    # Modified from https://github.com/Stick-To/CenterNet-tensorflow/blob/master/CenterNet.py _compute_one_image_loss
    slice_index = tf.argmin(ground_truth, axis=0)[0]
    ground_truth = tf.gather(ground_truth,
                             tf.range(0, slice_index, dtype=tf.int64))
    ngbbox_y = ground_truth[..., 0] / stride
    ngbbox_x = ground_truth[..., 1] / stride
    ngbbox_h = ground_truth[..., 2] / stride
    ngbbox_w = ground_truth[..., 3] / stride
    class_id = tf.cast(ground_truth[..., 4], dtype=tf.int32)
    ngbbox_yx = ground_truth[..., 0:2] / stride
    ngbbox_yx_round = tf.floor(ngbbox_yx)
    offset_gt = ngbbox_yx - ngbbox_yx_round
    size_gt = ground_truth[..., 2:4] / stride
    ngbbox_yx_round_int = tf.cast(ngbbox_yx_round, tf.int64)
    heatmap_loss = _heatmap_loss(heatmap, ngbbox_yx_round_int, ngbbox_y,
                                 ngbbox_x, ngbbox_h, ngbbox_w, class_id,
                                 meshgrid_y, meshgrid_x, pshape, num_classes)

    offset = tf.gather_nd(offset, ngbbox_yx_round_int)
    size = tf.gather_nd(size, ngbbox_yx_round_int)
    offset_loss = tf.reduce_mean(tf.abs(offset_gt - offset))
    size_loss = tf.reduce_mean(tf.abs(size_gt - size))
    offset_loss = tf.cond(tf.cast(slice_index, tf.bool), lambda: offset_loss,
                          lambda: tf.zeros_like(offset_loss, tf.float32))
    size_loss = tf.cond(tf.cast(slice_index, tf.bool), lambda: size_loss,
                        lambda: tf.zeros_like(size_loss, tf.float32))
    return heatmap_loss, size_loss, offset_loss


def loss(heatmap, size, offset, ground_truth, batch_size, num_classes):
    """loss

    Args:
        heatmap
        size
        offset
        ground_truth
        batch_size
        num_classes

    Returns:
        loss
    """
    pshape = [tf.shape(offset)[1], tf.shape(offset)[2]]
    h = tf.range(0., tf.cast(pshape[0], tf.float32), dtype=tf.float32)
    w = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
    [meshgrid_x, meshgrid_y] = tf.meshgrid(w, h)
    heatmap_loss = [None for _ in range(batch_size)]
    size_loss = [None for _ in range(batch_size)]
    offset_loss = [None for _ in range(batch_size)]
    for i in range(batch_size):
        single_loss = _cal_single_loss(heatmap[i, ...], size[i,
                                                             ...], offset[i,
                                                                          ...],
                                       ground_truth[i, ...], 4.0, pshape,
                                       num_classes, meshgrid_y, meshgrid_x)
        heatmap_loss[i] = single_loss[0]
        size_loss[i] = single_loss[1]
        offset_loss[i] = single_loss[2]

    heatmap_loss = tf.reduce_mean(tf.stack(heatmap_loss))
    size_loss = tf.reduce_mean(tf.stack(size_loss))
    offset_loss = tf.reduce_mean(tf.stack(offset_loss))
    total_loss = heatmap_loss + 0.1 * size_loss + offset_loss
    return heatmap_loss, size_loss, offset_loss, total_loss


def evaluation(heatmap, size, offset, score_threshold, top_K=100):
    """evaluation

    Args:
        heatmap
        size
        offset
        score_threshold
        top_K

    Returns:
        detection_pred
    """
    pshape = [tf.shape(offset)[1], tf.shape(offset)[2]]
    h = tf.range(0., tf.cast(pshape[0], tf.float32), dtype=tf.float32)
    w = tf.range(0., tf.cast(pshape[1], tf.float32), dtype=tf.float32)
    [meshgrid_x, meshgrid_y] = tf.meshgrid(w, h)

    heatmap = tf.sigmoid(heatmap)
    meshgrid_y = tf.expand_dims(meshgrid_y, axis=-1)
    meshgrid_x = tf.expand_dims(meshgrid_x, axis=-1)
    center = tf.concat([meshgrid_y, meshgrid_x], axis=-1)
    category = tf.expand_dims(tf.squeeze(
        tf.argmax(heatmap, axis=-1, output_type=tf.int32)),
                              axis=-1)
    meshgrid_xyz = tf.concat(
        [tf.zeros_like(category),
         tf.cast(center, tf.int32), category],
        axis=-1)
    keypoints = tf.gather_nd(heatmap, meshgrid_xyz)
    keypoints = tf.expand_dims(keypoints, axis=0)
    keypoints = tf.expand_dims(keypoints, axis=-1)

    keypoints_peak = tf.layers.max_pooling2d(inputs=keypoints,
                                             pool_size=3,
                                             strides=1,
                                             padding='same')
    keypoints_mask = tf.cast(tf.equal(keypoints, keypoints_peak), tf.float32)
    keypoints = keypoints * keypoints_mask
    scores = tf.reshape(keypoints, [-1])
    class_id = tf.reshape(category, [-1])
    bbox_yx = tf.reshape(center + offset, [-1, 2])
    bbox_hw = tf.reshape(size, [-1, 2])
    score_mask = scores > score_threshold
    scores = tf.boolean_mask(scores, score_mask)
    class_id = tf.boolean_mask(class_id, score_mask)
    bbox_yx = tf.boolean_mask(bbox_yx, score_mask)
    bbox_hw = tf.boolean_mask(bbox_hw, score_mask)
    bbox = tf.concat([bbox_yx - bbox_hw / 2., bbox_yx + bbox_hw / 2.],
                     axis=-1) * 4
    num_select = tf.cond(
        tf.shape(scores)[0] > top_K, lambda: top_K,
        lambda: tf.shape(scores)[0])
    select_scores, select_indices = tf.nn.top_k(scores, num_select)
    select_class_id = tf.gather(class_id, select_indices)
    select_bbox = tf.gather(bbox, select_indices)
    detection_pred = [select_scores, select_bbox, select_class_id]
    return detection_pred


def decode(detection):
    """decode

    Args:
        detection

    Returns:
        detection
    """
    # bbox type: two_points
    return np.hstack(
        (detection[1], detection[2].reshape(-1,
                                            1), detection[0].reshape(-1, 1)))
