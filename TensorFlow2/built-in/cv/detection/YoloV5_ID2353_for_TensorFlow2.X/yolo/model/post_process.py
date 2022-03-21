#! /usr/bin/env python
# coding=utf-8
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
#
# @Author: Longxing Tan, tanlongxing888@163.com

import npu_device
import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
from dataset.image_utils import xywh2xyxy, box_iou


def batch_non_max_suppression(prediction, conf_threshold=0.5, iou_threshold=0.25, classes=None, agnostic=False,
                              labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    prediction: batch_size * 3grid * (num_classes + 5)
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    num_classes = tf.shape(prediction)[-1] - 5
    candidates = prediction[..., 4] > conf_threshold
    output = [tf.zeros((0, 6))] * prediction.shape[0]

    for i, pred in enumerate(prediction):  # iter for image
        pred = pred[candidates[i]]  # filter by yolo confidence

        if not pred.shape[0]:
            continue

        box = xywh2xyxy(pred[:, :4])
        score = pred[:, 4]
        classes = tf.argmax(pred[..., 5:], axis=-1)

        pred_nms = []
        for clss in tf.unique(classes)[0]:
            mask = tf.math.equal(classes, clss)
            box_of_clss = tf.boolean_mask(box, mask)  # n_conf * 4
            classes_of_clss = tf.boolean_mask(classes, mask)  # n_conf
            score_of_clss = tf.boolean_mask(score, mask)  # n_conf

            select_indices = tf.image.non_max_suppression(box_of_clss, score_of_clss, max_output_size=50,
                                                          iou_threshold=iou_threshold)  # for one class
            box_of_clss = tf.gather(box_of_clss, select_indices)
            score_of_clss = tf.gather(tf.expand_dims(score_of_clss, -1), select_indices)
            classes_of_clss = tf.cast(tf.gather(tf.expand_dims(classes_of_clss, -1), select_indices), tf.float32)
            pred_of_clss = tf.concat([box_of_clss, score_of_clss, classes_of_clss], axis=-1)
            pred_nms.append(pred_of_clss)

        output[i] = tf.concat(pred_nms, axis=0)
    return output


def weighted_boxes_fusion():
    return
