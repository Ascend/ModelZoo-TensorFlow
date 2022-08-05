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
from src.constants import PARALLEL_ITERATIONS


def batch_non_max_suppression(
        boxes, scores,
        score_threshold, iou_threshold,
        max_boxes):
    """
    Arguments:
        boxes: a float tensor with shape [batch_size, N, 4].
        scores: a float tensor with shape [batch_size, N].
        score_threshold: a float number.
        iou_threshold: a float number, threshold for IoU.
        max_boxes: an integer, maximum number of retained boxes.
    Returns:
        boxes: a float tensor with shape [batch_size, max_boxes, 4].
        scores: a float tensor with shape [batch_size, max_boxes].
        num_detections: an int tensor with shape [batch_size].
    """
    def fn(x):
        boxes, scores = x

        # low scoring boxes are removed
        ids = tf.where(tf.greater_equal(scores, score_threshold))
        ids = tf.squeeze(ids, axis=1)
        boxes = tf.gather(boxes, ids)
        scores = tf.gather(scores, ids)

        selected_indices = tf.image.non_max_suppression(
            boxes, scores, max_boxes, iou_threshold
        )
        boxes = tf.gather(boxes, selected_indices)
        scores = tf.gather(scores, selected_indices)
        num_boxes = tf.to_int32(tf.shape(boxes)[0])

        zero_padding = max_boxes - num_boxes
        boxes = tf.pad(boxes, [[0, zero_padding], [0, 0]])
        scores = tf.pad(scores, [[0, zero_padding]])

        boxes.set_shape([max_boxes, 4])
        scores.set_shape([max_boxes])
        return boxes, scores, num_boxes

    boxes, scores, num_detections = tf.map_fn(
        fn, [boxes, scores],
        dtype=(tf.float32, tf.float32, tf.int32),
        parallel_iterations=PARALLEL_ITERATIONS,
        back_prop=False, swap_memory=False, infer_shape=True
    )
    return boxes, scores, num_detections
