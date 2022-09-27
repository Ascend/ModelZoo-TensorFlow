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
import math

import numpy as np
import tensorflow as tf


class BBoxUtility(object):
    def __init__(self, num_classes, rpn_pre_boxes=12000, rpn_nms=0.7, nms_iou=0.3, min_k=300):
        self.num_classes = num_classes
        self.rpn_pre_boxes = rpn_pre_boxes
        self.rpn_nms = rpn_nms
        self.nms_iou = nms_iou
        self._min_k = min_k

    def decode_boxes(self, mbox_loc, anchors, variances):
        # 获得先验框的宽与高
        anchor_width = anchors[:, 2] - anchors[:, 0]
        anchor_height = anchors[:, 3] - anchors[:, 1]
        # 获得先验框的中心点
        anchor_center_x = 0.5 * (anchors[:, 2] + anchors[:, 0])
        anchor_center_y = 0.5 * (anchors[:, 3] + anchors[:, 1])

        # 真实框距离先验框中心的xy轴偏移情况
        detections_center_x = mbox_loc[:, 0] * anchor_width * variances[0]
        detections_center_x += anchor_center_x
        detections_center_y = mbox_loc[:, 1] * anchor_height * variances[1]
        detections_center_y += anchor_center_y

        # 真实框的宽与高的求取
        detections_width = np.exp(mbox_loc[:, 2] * variances[2])
        detections_width *= anchor_width
        detections_height = np.exp(mbox_loc[:, 3] * variances[3])
        detections_height *= anchor_height

        # 获取真实框的左上角与右下角
        detections_xmin = detections_center_x - 0.5 * detections_width
        detections_ymin = detections_center_y - 0.5 * detections_height
        detections_xmax = detections_center_x + 0.5 * detections_width
        detections_ymax = detections_center_y + 0.5 * detections_height

        # 真实框的左上角与右下角进行堆叠
        detections = np.concatenate((detections_xmin[:, None],
                                     detections_ymin[:, None],
                                     detections_xmax[:, None],
                                     detections_ymax[:, None]), axis=-1)
        # 防止超出0与1
        detections = np.minimum(np.maximum(detections, 0.0), 1.0)
        return detections

    def detection_out_rpn(self, predictions, anchors, variances=[0.25, 0.25, 0.25, 0.25]):
        mbox_conf = predictions[0]
        mbox_loc = predictions[1]

        results = []
        for i in range(len(mbox_loc)):
            detections = self.decode_boxes(mbox_loc[i], anchors, variances)
            c_confs = mbox_conf[i, :, 0]
            c_confs_argsort = np.argsort(c_confs)[::-1][:self.rpn_pre_boxes]

            confs_to_process = c_confs[c_confs_argsort]
            boxes_to_process = detections[c_confs_argsort, :]
            boxes_to_process = boxes_to_process.astype('float32')
            idx = tf.image.non_max_suppression(boxes_to_process, confs_to_process, self._min_k,
                                               iou_threshold=self.rpn_nms).numpy()

            good_boxes = boxes_to_process[idx]
            results.append(good_boxes)
        return np.array(results)

    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def detection_out_classifier(self, predictions, rpn_results, image_shape, input_shape, confidence=0.5,
                                 variances=[0.125, 0.125, 0.25, 0.25]):
        proposal_conf = predictions[0]
        proposal_loc = predictions[1]

        results = []
        for i in range(len(proposal_conf)):
            results.append([])
            detections = []
            rpn_results[i, :, 2] = rpn_results[i, :, 2] - rpn_results[i, :, 0]
            rpn_results[i, :, 3] = rpn_results[i, :, 3] - rpn_results[i, :, 1]
            rpn_results[i, :, 0] = rpn_results[i, :, 0] + rpn_results[i, :, 2] / 2
            rpn_results[i, :, 1] = rpn_results[i, :, 1] + rpn_results[i, :, 3] / 2
            for j in range(proposal_conf[i].shape[0]):
                score = np.max(proposal_conf[i][j, :-1])
                label = np.argmax(proposal_conf[i][j, :-1])
                if score < confidence:
                    continue
                x, y, w, h = rpn_results[i, j, :]
                tx, ty, tw, th = proposal_loc[i][j, 4 * label: 4 * (label + 1)]

                x1 = tx * variances[0] * w + x
                y1 = ty * variances[1] * h + y
                w1 = math.exp(tw * variances[2]) * w
                h1 = math.exp(th * variances[3]) * h

                xmin = x1 - w1 / 2.
                ymin = y1 - h1 / 2.
                xmax = x1 + w1 / 2
                ymax = y1 + h1 / 2

                detections.append([xmin, ymin, xmax, ymax, score, label])

            detections = np.array(detections)
            if len(detections) > 0:
                for c in range(self.num_classes):
                    c_confs_m = detections[:, -1] == c
                    if len(detections[c_confs_m]) > 0:
                        boxes_to_process = detections[:, :4][c_confs_m]
                        confs_to_process = detections[:, 4][c_confs_m]
                        boxes_to_process = boxes_to_process.astype('float32')
                        idx = tf.image.non_max_suppression(boxes_to_process, confs_to_process, self._min_k,
                                                           iou_threshold=self.nms_iou).numpy()
                        results[-1].extend(detections[c_confs_m][idx])

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4]) / 2, results[-1][:, 2:4] - results[-1][:,
                                                                                                        0:2]
                results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)

        return results
