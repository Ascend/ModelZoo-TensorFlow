#! /usr/bin/env python
# coding=utf-8

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

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
import time

class Dataset(object):
    """implement Dataset here"""

    def __init__(self, FLAGS, is_training: bool, dataset_type: str = "converted_coco"):
        self.tiny = FLAGS.tiny
        self.strides, self.anchors, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        self.dataset_type = dataset_type

        self.annot_path = (
            FLAGS.data_path + '/train2017.txt' if is_training else FLAGS.anno_converted
        )
        self.input_sizes = (
            cfg.TRAIN.INPUT_SIZE if is_training else cfg.TEST.INPUT_SIZE
        )
        self.batch_size = (
            FLAGS.batch_size if is_training else cfg.TEST.BATCH_SIZE
        )
        self.data_aug = cfg.TRAIN.DATA_AUG if is_training else False

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 450

        self.annotations = self.load_annotations(FLAGS)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        self.mosaic = FLAGS.mosaic and is_training

        self._data_buff = dict()

    def load_annotations(self,FLAGS):
        with open(self.annot_path, "r") as f:
            txt = f.readlines()
            if self.dataset_type == "converted_coco":
                annotations = [
                    line.strip()
                    for line in txt
                    if len(line.strip().split()[1:]) != 0
                ]
            elif self.dataset_type == "yolo":
                annotations = []
                for line in txt:
                    image_path = line.strip()
                    root, _ = os.path.splitext(image_path)
                    with open(root + ".txt") as fd:
                        boxes = fd.readlines()
                        string = ""
                        for box in boxes:
                            box = box.strip()
                            box = box.split()
                            class_num = int(box[0])
                            center_x = float(box[1])
                            center_y = float(box[2])
                            half_width = float(box[3]) / 2
                            half_height = float(box[4]) / 2
                            string += " {},{},{},{},{}".format(
                                center_x - half_width,
                                center_y - half_height,
                                center_x + half_width,
                                center_y + half_height,
                                class_num,
                            )
                        annotations.append(image_path + string)

        np.random.shuffle(annotations)

        # shard
        if FLAGS.rank_size > 1:
            len_annotations = len(annotations)
            annotations = annotations[int(len_annotations//int(FLAGS.rank_size))*FLAGS.rank:int(len_annotations//int(FLAGS.rank_size))*(FLAGS.rank+1)]

        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        num = 0
        if self.batch_count < self.num_batchs:
            annotations = []
            while num < self.batch_size:
                index = self.batch_count * self.batch_size + num
                if index >= self.num_samples:
                    index -= self.num_samples
                annotation = self.annotations[index]
                annotations.append(annotation)
                num += 1
            self.batch_count += 1
            return annotations
        else:
            raise StopIteration

    def __len__(self):
        return self.num_batchs

    def rewind(self):
        self.batch_count = 0
        np.random.shuffle(self.annotations)

###################################

class DatasetFetcher:

    def __init__(self, p):
        self.batch_size = p.batch_size
        self.anchor_per_scale = p.anchor_per_scale
        self.num_classes = p.num_classes
        self.max_bbox_per_scale = p.max_bbox_per_scale
        self.dataset_type = p.dataset_type
        self.data_aug = p.data_aug
        self.strides = p.strides
        self.anchors = p.anchors
        self.annotations = p.annotations
        self.mosaic = p.mosaic

        self.train_input_size = cfg.TRAIN.INPUT_SIZE
        self.mosaic_border = [-self.train_input_size // 4, -self.train_input_size // 4] # keep image size
        self.train_output_sizes = self.train_input_size // self.strides

        self.batch_image = np.zeros(
            (
                self.batch_size,
                self.train_input_size,
                self.train_input_size,
                3,
            ),
            dtype=np.float32,
        )

        self.batch_label_sbbox = np.zeros(
            (
                self.batch_size,
                self.train_output_sizes[0],
                self.train_output_sizes[0],
                self.anchor_per_scale,
                5 + self.num_classes,
            ),
            dtype=np.float32,
        )
        self.batch_label_mbbox = np.zeros(
            (
                self.batch_size,
                self.train_output_sizes[1],
                self.train_output_sizes[1],
                self.anchor_per_scale,
                5 + self.num_classes,
            ),
            dtype=np.float32,
        )
        self.batch_label_lbbox = np.zeros(
            (
                self.batch_size,
                self.train_output_sizes[2],
                self.train_output_sizes[2],
                self.anchor_per_scale,
                5 + self.num_classes,
            ),
            dtype=np.float32,
        )

        self.batch_sbboxes = np.zeros(
            (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
        )
        self.batch_mbboxes = np.zeros(
            (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
        )
        self.batch_lbboxes = np.zeros(
            (self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32
        )
        self.label = [
            np.zeros(
                (
                    self.train_output_sizes[i],
                    self.train_output_sizes[i],
                    self.anchor_per_scale,
                    5 + self.num_classes,
                )
            )
            for i in range(3)
        ]
        self.bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]

    def process_annotations(self, annotations):
        batch_image = self.batch_image
        batch_label_sbbox = self.batch_label_sbbox
        batch_label_mbbox = self.batch_label_mbbox
        batch_label_lbbox = self.batch_label_lbbox
        batch_sbboxes = self.batch_sbboxes
        batch_mbboxes = self.batch_mbboxes
        batch_lbboxes = self.batch_lbboxes
        
        batch_image.fill(0)
        batch_label_sbbox.fill(0)
        batch_label_mbbox.fill(0)
        batch_label_lbbox.fill(0)
        batch_sbboxes.fill(0)
        batch_mbboxes.fill(0)
        batch_lbboxes.fill(0)
        
        num = 0
        for annotation in annotations:
            image, bboxes, image_path, scale, dw, dh = self.load_img_and_labels(annotation)
            (
                label_sbbox,
                label_mbbox,
                label_lbbox,
                sbboxes,
                mbboxes,
                lbboxes,
            ) = self.preprocess_true_boxes(bboxes)

            batch_image[num, :, :, :] = image
            batch_image_id = int(image_path.split('/')[-1].split('.')[0][-12:])
            batch_label_sbbox[num, :, :, :, :] = label_sbbox
            batch_label_mbbox[num, :, :, :, :] = label_mbbox
            batch_label_lbbox[num, :, :, :, :] = label_lbbox
            batch_sbboxes[num, :, :] = sbboxes
            batch_mbboxes[num, :, :] = mbboxes
            batch_lbboxes[num, :, :] = lbboxes
            num += 1
        batch_smaller_target = batch_label_sbbox, batch_sbboxes
        batch_medium_target = batch_label_mbbox, batch_mbboxes
        batch_larger_target = batch_label_lbbox, batch_lbboxes

        return (
            batch_image,
            (
                batch_smaller_target,
                batch_medium_target,
                batch_larger_target,
            ),
            batch_image_id,
            scale,
            dw,
            dh,
        )

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(
                0, int(max_bbox[0] - random.uniform(0, max_l_trans))
            )
            crop_ymin = max(
                0, int(max_bbox[1] - random.uniform(0, max_u_trans))
            )
            crop_xmax = max(
                w, int(max_bbox[2] + random.uniform(0, max_r_trans))
            )
            crop_ymax = max(
                h, int(max_bbox[3] + random.uniform(0, max_d_trans))
            )

            image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate(
                [
                    np.min(bboxes[:, 0:2], axis=0),
                    np.max(bboxes[:, 2:4], axis=0),
                ],
                axis=-1,
            )

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-max_l_trans, max_r_trans)
            ty = random.uniform(-max_u_trans, max_d_trans)
#            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
#            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def mosaic_4(self, image, bboxes):
        """
        reference:ultralytics/yolov5/utils/dataset.py
        """
        bboxes4 = []
        s = int(self.train_input_size / 2)
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]
        imgs = [(np.copy(image), np.copy(bboxes))] + [(self.parse_annotation(annotation)[0], self.parse_annotation(annotation)[1]) for annotation in random.choices(self.annotations, k=3)]
        for i, (image, bboxes) in enumerate(imgs):
            h, w = image.shape[0], image.shape[1]
            if i == 0: # top left
                img4 = np.full((s * 2, s * 2, image.shape[2]), 114, dtype=np.uint8) # base image
                x1a, y1a, x2a, y2a = max(xc -w, 0), max(yc - h, 0), xc, yc # large image
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h # small image
            elif i == 1: # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2: # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else: # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            img4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # move bboxes to large image
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + padw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + padh

            # clip bbox
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], x1a, x2a)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], y1a, y2a)

            # desert useless bboxes
            mask = np.where((bboxes[:, 2] - bboxes[:, 0] < 3) | (bboxes[:, 3] - bboxes[:, 1] < 3))
            bboxes = np.delete(bboxes, mask, axis=0)
            bboxes4.append(bboxes)
        bboxes4 = np.concatenate(bboxes4, axis=0)
        if bboxes4.shape[0] == 0:
            return self.mosaic_4(image, bboxes)
        return img4, bboxes4

    def parse_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = cv2.imread(image_path)
        if self.dataset_type == "converted_coco":
            bboxes = np.array(
                [list(map(float, box.split(","))) for box in line[1:]]
            )
        elif self.dataset_type == "yolo":
            height, width, _ = image.shape
            bboxes = np.array(
                [list(map(float, box.split(","))) for box in line[1:]]
            )
            bboxes = bboxes * np.array([width, height, width, height, 1])
            bboxes = bboxes.astype(np.int64)
        return image, bboxes, image_path

    def load_img_and_labels(self, annotation):
        image, bboxes, image_path = self.parse_annotation(annotation)
        
        if self.mosaic:
            image, bboxes = self.mosaic_4(image, bboxes)
        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(
                np.copy(image), np.copy(bboxes)
            )
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(
                np.copy(image), np.copy(bboxes)
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, bboxes, scale, dw, dh = utils.image_preprocess(
            np.copy(image),
            [self.train_input_size, self.train_input_size],
            np.copy(bboxes),
        )
        return image, bboxes, image_path, scale, dw, dh


    def preprocess_true_boxes(self, bboxes, anchors_t=4.0):
        label = self.label
        bboxes_xywh = self.bboxes_xywh
        for l in label:
            l.fill(0)
        for b in bboxes_xywh:
            b.fill(0)
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4].copy()
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[int(bbox_class_ind)] = 1.0
            uniform_distribution = np.full(
                self.num_classes, 1.0 / self.num_classes
            )
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate(
                [
                    (bbox_coor[2:].copy() + bbox_coor[:2].copy()) * 0.5,
                    bbox_coor[2:].copy() - bbox_coor[:2].copy(),
                ],
                axis=-1,
            )
            bbox_xywh_scaled = (bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis])
            utils.bbox_iou_(bbox_xywh_scaled, bbox_xywh_scaled)
            for i in range(3):
                r = (bbox_xywh[np.newaxis, 2:].copy() / self.anchors[i]) # wp/wa hp/ha
                mask = np.array([False, False, False])
                for ind in range(3):
                    if np.all(r[ind, :].copy() < anchors_t) and np.all(r[ind, :] > (1 / anchors_t)):
                        mask[ind] = True
                if np.any(mask):
                    x, y = bbox_xywh_scaled[i, :2]
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    locations = [(xind, yind)]
                    if x >= 1 and x % 1 < 0.5:
                        locations.append((xind - 1, yind))
                    if y >= 1 and y % 1 < 0.5:
                        locations.append((xind, yind - 1))
                    for xind, yind in locations:
                        label[i][yind, xind, mask, :] = 0
                        label[i][yind, xind, mask, 0:4] = bbox_xywh
                        label[i][yind, xind, mask, 4:5] = 1.0
                        label[i][yind, xind, mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

        label_sbbox, label_mbbox, label_lbbox = label[0].copy(), label[1].copy(), label[2].copy()
        sbboxes, mbboxes, lbboxes = bboxes_xywh[0].copy(), bboxes_xywh[1].copy(), bboxes_xywh[2].copy()
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes