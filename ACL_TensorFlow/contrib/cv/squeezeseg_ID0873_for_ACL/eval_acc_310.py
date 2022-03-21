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
# Author: Bichen Wu (bichen@berkeley.edu) 03/07/2017


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from six.moves import xrange

BATCH_SIZE = 1
num_images = len(os.listdir("./bin_out/lidar"))
input = os.listdir("./bin_out/lidar")
input_mask = os.listdir("./bin_out/lidar_mask")
pred_cls_total = os.listdir(os.path.join("./bin_out/pred_cls", os.listdir("./bin_out/pred_cls")[0]))
input_label = os.listdir("./bin_out/label")
input = sorted(input, key=lambda x: int(x[0:5]))
input_mask = sorted(input_mask, key=lambda x: int(x[0:5]))
pred_cls_total = sorted(pred_cls_total, key=lambda x: int(x[0:5]))
input_label = sorted(input_label, key=lambda x: int(x[0:5]))

NUM_CLASS = 4
tp_sum = 0
fn_sum = 0
fp_sum = 0
CLASSES = ['unknown', 'car', 'pedestrian', 'cyclist']
DENOM_EPSILON = 1e-12


def evaluate_iou(label, pred, n_class, epsilon=1e-12):
    """Evaluation script to compute pixel level IoU.
  Args:
    label: N-d array of shape [batch, W, H], where each element is a class
        index.
    pred: N-d array of shape [batch, W, H], the each element is the predicted
        class index.
    n_class: number of classes
    epsilon: a small value to prevent division by 0

  Returns:
    IoU: array of lengh n_class, where each element is the average IoU for this
        class.
    tps: same shape as IoU, where each element is the number of TP for each
        class.
    fps: same shape as IoU, where each element is the number of FP for each
        class.
    fns: same shape as IoU, where each element is the number of FN for each
        class.
  """

    assert label.shape == pred.shape, \
        'label and pred shape mismatch: {} vs {}'.format(
            label.shape, pred.shape)
    ious = np.zeros(n_class)
    tps = np.zeros(n_class)
    fns = np.zeros(n_class)
    fps = np.zeros(n_class)
    for cls_id in range(n_class):
        tp = np.sum(pred[label == cls_id] == cls_id)
        fp = np.sum(label[pred == cls_id] != cls_id)
        fn = np.sum(pred[label == cls_id] != cls_id)
        ious[cls_id] = tp / (tp + fn + fp + epsilon)
        tps[cls_id] = tp
        fps[cls_id] = fp
        fns[cls_id] = fn
    return ious, tps, fps, fns


for i in xrange(int(num_images / BATCH_SIZE)):
    offset = max((i + 1) * BATCH_SIZE - num_images, 0)

    lidar_mask_per_batch = np.fromfile(
        os.path.join("./bin_out/lidar_mask", input_mask[i]),
        dtype=np.float32).reshape(1, 64, 512, 1)

    pred_cls = np.fromfile(
        os.path.join(os.path.join("./bin_out/pred_cls", os.listdir("./bin_out/pred_cls")[0]), pred_cls_total[i]),
        dtype=np.int32).reshape(1, 64, 512)

    label_per_batch = np.fromfile(
        os.path.join("./bin_out/label", input_label[i]),
        dtype=np.int32).reshape(1, 64, 512)

    iou, tps, fps, fns = evaluate_iou(
        label_per_batch[:BATCH_SIZE - offset],
        pred_cls[:BATCH_SIZE - offset] \
        * np.squeeze(lidar_mask_per_batch[:BATCH_SIZE - offset]),
        NUM_CLASS)

    tp_sum += tps
    fn_sum += fns
    fp_sum += fps

ious = tp_sum.astype(np.float64) / (tp_sum + fn_sum + fp_sum + DENOM_EPSILON)
pr = tp_sum.astype(np.float64) / (tp_sum + fp_sum + DENOM_EPSILON)
re = tp_sum.astype(np.float64) / (tp_sum + fn_sum + DENOM_EPSILON)
print(pr)
for i in range(1, NUM_CLASS):
    print('    {}:'.format(CLASSES[i]))
    print('\tPixel-seg: P: {:.3f}, R: {:.3f}, IoU: {:.3f}'.format(
        pr[i], re[i], ious[i]))
