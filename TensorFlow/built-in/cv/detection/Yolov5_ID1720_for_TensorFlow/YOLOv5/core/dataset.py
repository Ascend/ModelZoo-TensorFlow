#! /usr/bin/env python
# -*- coding: utf-8 -*-
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
# ==============================================================================
import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg

class Dataset(object):
    def __init__(self, dataset_type, net_type, shard_num, shard_id):
        self.net_type = net_type
        self.shard_num = shard_num
        self.shard_id = shard_id
        self.annot_path = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else [cfg.TEST.INPUT_SIZE]
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug = cfg.TRAIN.DATA_AUG if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations()
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        shard_size = int(np.floor(len(annotations) / self.shard_num))
        annotations = annotations[(self.shard_id*shard_size):((self.shard_id+1)*shard_size)]

        np.random.shuffle(annotations)
        return annotations

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            num = 0
            if self.batch_count < self.num_batchs:
                batch_annotations = []
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples

                    annotation = self.annotations[index]
                    batch_annotations.append(annotation)

                    num += 1
                
                self.batch_count += 1

                return batch_annotations            
            else:
                raise StopIteration

    def __len__(self):
        return self.num_batchs

    def rewind(self):
        self.batch_count = 0
        np.random.shuffle(self.annotations)

###########################################################################

class DatasetBatchFetcher(object):
    def __init__(self, p):
        self.p = p

        self.annotations = p.annotations

        self.net_type = p.net_type
        self.batch_size = p.batch_size
        self.data_aug = p.data_aug

        self.input_sizes = p.input_sizes
        self.strides = p.strides
        self.num_classes = p.num_classes
        self.anchors = p.anchors
        self.anchor_per_scale = p.anchor_per_scale
        self.max_bbox_per_scale = p.max_bbox_per_scale

        self.data_buffs = dict()
        self.batch_annotations = []

    def process(self):
        self.input_size = random.choice(self.input_sizes)
        self.output_sizes = self.input_size // self.strides
            
        if not self.input_size in self.data_buffs.keys():
            class DataBuff:
                pass

            self.data_buffs[self.input_size] = DataBuff()
            data_buff = self.data_buffs[self.input_size]

            data_buff.batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3))

            if self.net_type == 'tiny':
                data_buff.batch_label_mbbox = np.zeros((self.batch_size, self.output_sizes[0], self.output_sizes[0],
                                              self.anchor_per_scale, 5 + self.num_classes))
                data_buff.batch_label_lbbox = np.zeros((self.batch_size, self.output_sizes[1], self.output_sizes[1],
                                              self.anchor_per_scale, 5 + self.num_classes))

                data_buff.batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
                data_buff.batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            
            else:
                data_buff.batch_label_sbbox = np.zeros((self.batch_size, self.output_sizes[0], self.output_sizes[0],
                                              self.anchor_per_scale, 5 + self.num_classes))
                data_buff.batch_label_mbbox = np.zeros((self.batch_size, self.output_sizes[1], self.output_sizes[1],
                                              self.anchor_per_scale, 5 + self.num_classes))
                data_buff.batch_label_lbbox = np.zeros((self.batch_size, self.output_sizes[2], self.output_sizes[2],
                                              self.anchor_per_scale, 5 + self.num_classes))

                data_buff.batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
                data_buff.batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
                data_buff.batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

        data_buff = self.data_buffs[self.input_size]
        batch_image = data_buff.batch_image
        batch_image.fill(0)
        if self.net_type == 'tiny':
            batch_label_mbbox = data_buff.batch_label_mbbox
            batch_label_lbbox = data_buff.batch_label_lbbox

            batch_mbboxes = data_buff.batch_mbboxes
            batch_lbboxes = data_buff.batch_lbboxes

            batch_label_mbbox.fill(0)
            batch_label_lbbox.fill(0)
            batch_mbboxes.fill(0)
            batch_lbboxes.fill(0)
        else:
            batch_label_sbbox = data_buff.batch_label_sbbox
            batch_label_mbbox = data_buff.batch_label_mbbox
            batch_label_lbbox = data_buff.batch_label_lbbox

            batch_sbboxes = data_buff.batch_sbboxes
            batch_mbboxes = data_buff.batch_mbboxes
            batch_lbboxes = data_buff.batch_lbboxes

            batch_label_sbbox.fill(0)
            batch_label_mbbox.fill(0)
            batch_label_lbbox.fill(0)
            batch_sbboxes.fill(0)
            batch_mbboxes.fill(0)
            batch_lbboxes.fill(0)

        self.data_buff = data_buff

        batch_image_id = np.zeros(self.batch_size)
        batch_scale = np.zeros(self.batch_size)
        batch_dw = np.zeros(self.batch_size)
        batch_dh = np.zeros(self.batch_size)
        batch_image_path = []

        num = 0
        for annotation in self.batch_annotations:
            image, bboxes, image_id, scale, dw, dh, image_path = self.parse_annotation(annotation)

            batch_image[num, :, :, :] = image
            batch_image_id[num] = image_id
            batch_scale[num] = scale
            batch_dw[num] = dw
            batch_dh[num] = dh
            batch_image_path.append(image_path)

            if self.net_type == 'tiny':
                label_mbbox, label_lbbox, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                
                batch_label_mbbox[num, :, :, :, :] = label_mbbox
                batch_label_lbbox[num, :, :, :, :] = label_lbbox
                batch_mbboxes[num, :, :] = mbboxes
                batch_lbboxes[num, :, :] = lbboxes
            else:
                label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                batch_label_sbbox[num, :, :, :, :] = label_sbbox
                batch_label_mbbox[num, :, :, :, :] = label_mbbox
                batch_label_lbbox[num, :, :, :, :] = label_lbbox
                batch_sbboxes[num, :, :] = sbboxes
                batch_mbboxes[num, :, :] = mbboxes
                batch_lbboxes[num, :, :] = lbboxes

            num += 1

        if self.net_type == 'tiny':
            return batch_image, batch_label_mbbox, batch_label_lbbox, batch_mbboxes, batch_lbboxes, \
                                batch_image_id, batch_scale, batch_dw, batch_dh, batch_image_path
        else:
            return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                                batch_sbboxes, batch_mbboxes, batch_lbboxes, \
                                batch_image_id, batch_scale, batch_dw, batch_dh, batch_image_path


    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]
        return image, bboxes


    def random_vertical_flip(self, image, bboxes):
        if random.random() < 0.5:
            h, _, _ = image.shape
            image = image[::-1, :, :]
            bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]
        return image, bboxes


    def random_horizontal_vertical_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

            h, _, _ = image.shape
            image = image[::-1, :, :]
            bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]
        return image, bboxes


    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
        return image, bboxes


    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
        return image, bboxes


    def parse_annotation(self, annotation):
        image, image_id, bboxes, image_path = self.__read_annotation(annotation)
        image, bboxes, image_id, scale, dw, dh, image_path = self.__after_read_annotation(image, image_id, bboxes, image_path)

        return image, bboxes, image_id, scale, dw, dh, image_path


    def __read_annotation(self, annotation):
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        
        image = cv2.imread(image_path)
        image_id = int(image_path.split('/')[-1].split('.')[0][-12:])
        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])
        if len(bboxes) <= 0:
            raise KeyError("bboxes.len=%d does not exist ... " % len(bboxes))

        return image, image_id, bboxes, image_path


    def __after_read_annotation(self, image, image_id, bboxes, image_path):
        if self.data_aug:
            image, bboxes = self.mosaic_4(image, np.copy(bboxes))

            image, bboxes = self.random_horizontal_flip(image, np.copy(bboxes))
            #image, bboxes = self.random_vertical_flip(image, np.copy(bboxes))
            #image, bboxes = self.random_horizontal_vertical_flip(image, np.copy(bboxes))
            #image, bboxes = self.random_crop(image, np.copy(bboxes))
            #image, bboxes = self.random_translate(image, np.copy(bboxes))

        image, bboxes, scale, dw, dh = utils.image_preporcess(image, [self.input_size, self.input_size], np.copy(bboxes))

        return image, bboxes, image_id, scale, dw, dh, image_path


    def mosaic_4(self, image, bboxes):
        scale_ratio = random.uniform(0.8, 1.25)
        self.mosaic_border = [-self.input_size * scale_ratio // 4,
                              -self.input_size * scale_ratio // 4]

        bboxes4 = []
        s = int(self.input_size * scale_ratio / 2)
        yc, xc = [int(random.uniform(-x, 2*s+x)) for x in self.mosaic_border]  # mosaic center

        imgs = [(np.copy(image), np.copy(bboxes))]
        for annotation in random.choices(self.annotations, k=3):
            ret = self.__read_annotation(annotation)
            imgs.append((ret[0], ret[2]))

        for i, (image, bboxes) in enumerate(imgs):
            raw_bboxes = np.copy(bboxes)

            h, w = image.shape[0], image.shape[1]
            if i==0 :  # top left
                img4 = np.full((s*2, s*2, image.shape[2]), 114, dtype=np.uint8)  # base image
                x1a, y1a, x2a, y2a = max(xc-w, 0), max(yc-h, 0), xc, yc  # in large image
                x1b, y1b, x2b, y2b = w-(x2a-x1a), h-(y2a-y1a), w, h  # in small image
            elif i==1 :  # top right
                x1a, y1a, x2a, y2a = xc, max(yc-h, 0), min(xc+w, s*2), yc
                x1b, y1b, x2b, y2b = 0, h-(y2a-y1a), min(w, x2a-x1a), h
            elif i==2 :  # bottom left
                x1a, y1a, x2a, y2a = max(xc-w, 0), yc, xc, min(s*2, yc+h)
                x1b, y1b, x2b, y2b = w-(x2a-x1a), 0, w, min(y2a-y1a, h)
            elif i==3 :  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc+w, s*2), min(s*2, yc+h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a-x1a), min(y2a-y1a, h)
            
            img4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw, padh = x1a-x1b, y1a-y1b

            # translate bboxes to large image
            bboxes[:, [0,2]] += padw
            bboxes[:, [1,3]] += padh

            # clip bboxes
            bboxes[:, [0,2]] = np.clip(bboxes[:, [0,2]], x1a, x2a-1)
            bboxes[:, [1,3]] = np.clip(bboxes[:, [1,3]], y1a, y2a-1)

            # desert useless bboxes
            del_mask = np.where( (bboxes[:, 2] - bboxes[:, 0] < 3) | (bboxes[:, 3] - bboxes[:, 1] < 3)
                               | ( (bboxes[:, 2] - bboxes[:, 0] < (raw_bboxes[:, 2] - raw_bboxes[:, 0]) * 0.2)
                                 | (bboxes[:, 3] - bboxes[:, 1] < (raw_bboxes[:, 3] - raw_bboxes[:, 1]) * 0.2)
                               ) )
            bboxes = np.delete(bboxes, del_mask, axis=0)
            bboxes4.append(bboxes)
        
        bboxes4 = np.concatenate(bboxes4, axis=0)

        if bboxes4.shape[0] == 0:
            return self.mosaic_4(image, bboxes)
        
        return img4, bboxes4


    def bbox_iou(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5, boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5, boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        return inter_area / union_area


    def preprocess_true_boxes(self, bboxes):
        anchor_size = 3
        if self.net_type == 'tiny':
            anchor_size = 2

        data_buff = self.data_buff
        if not hasattr(data_buff, 'label'):
            data_buff.label = [np.zeros((self.output_sizes[i], self.output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(anchor_size)]
            data_buff.bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(anchor_size)]

        label       = data_buff.label
        bboxes_xywh = data_buff.bboxes_xywh

        for l in label:
            l.fill(0)
        for b in bboxes_xywh:
            b.fill(0)

        bbox_count = np.zeros((anchor_size,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False            
            for i in range(anchor_size):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1
                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        
        if self.net_type == 'tiny':
            label_mbbox, label_lbbox = label
            mbboxes, lbboxes = bboxes_xywh
            return label_mbbox, label_lbbox, mbboxes, lbboxes
        else:
            label_sbbox, label_mbbox, label_lbbox = label
            sbboxes, mbboxes, lbboxes = bboxes_xywh
            return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

