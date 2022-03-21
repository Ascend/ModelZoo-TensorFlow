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
from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# NPU options
__C.NPU = edict()
# 'allow_fp32_to_fp16' 'force_fp16' 'must_keep_origin_dtype' 'allow_mix_precision'
__C.NPU.PRECISION_MODE = 'allow_mix_precision'
__C.NPU.LOSS_SCALE_FLAG = 1
__C.NPU.LOSS_SCALE = 256
__C.NPU.OVERFLOW_DUMP = False

# YOLO options
__C.YOLO = edict()


# Set the class name
__C.YOLO.NET_TYPE = 'darknet53' # 'darknet53' 'mobilenetv2' 'mobilenetv3' 'mobilenetv3_small'
__C.YOLO.CLASSES = '/npu/traindata/coco2014/labels.txt'
#__C.YOLO.ANCHORS = 'data/anchors/basline_anchors.txt' # yolov3/5 : yolo_anchors.txt; yolov4 : yolov4_anchors.txt
__C.YOLO.ANCHORS = 'data/anchors/yolo_anchors.txt'
__C.YOLO.MOVING_AVE_DECAY = 0.9995
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.STRIDES_TINY = [16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5
__C.YOLO.UPSAMPLE_METHOD = 'resize'

__C.YOLO.WIDTH_SCALE_V5 = 0.50 # yolov5 small:0.50 / middle:0.75 / large:1.00 / extend:1.25
__C.YOLO.DEPTH_SCALE_V5 = 0.33 # yolov5 small:0.33(1/3) / middle:0.67(2/3) / large:1.00 / extend:1.33(4/3)

__C.YOLO.ORIGINAL_WEIGHT = None
__C.YOLO.DEMO_WEIGHT = None


# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = '/npu/traindata/coco2014/train_annotation.txt'
#__C.TRAIN.BATCH_SIZE = 2 if __C.YOLO.NET_TYPE == 'darknet53' else 8
__C.TRAIN.BATCH_SIZE = 8
__C.TRAIN.LEARN_RATE_INIT = 5e-4
__C.TRAIN.LEARN_RATE_END = 5e-6

#__C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608] if not 'mobilenetv3' in __C.YOLO.NET_TYPE else [416]
__C.TRAIN.INPUT_SIZE = [640]
__C.TRAIN.DATA_AUG = True
__C.TRAIN.ADAM_BETA1 = 0.9
__C.TRAIN.ADAM_BETA2 = 0.999
__C.TRAIN.ADAM_EPSILON = 1e-8
__C.TRAIN.WEIGHT_L2_REG_COEF = 0.0005
__C.TRAIN.WARMUP_EPOCHS = 3
__C.TRAIN.FIRST_STAGE_EPOCHS = 100
__C.TRAIN.SECOND_STAGE_EPOCHS = 300 # 1000
__C.TRAIN.MAX_TOTAL_STEPS = 1e8
__C.TRAIN.INITIAL_WEIGHT = None
__C.TRAIN.CKPT_PATH = 'ckpts'


# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = '/npu/traindata/coco2014/val_annotation.txt'
__C.TEST.ANNOT_PATH_ = '/npu/traindata/coco2014/annotations/instances_val2014.json'
__C.TEST.BATCH_SIZE = 1
__C.TEST.INPUT_SIZE = 640
__C.TEST.MAX_STEPS = 1e8
__C.TEST.DATA_AUG = False
__C.TEST.WRITE_IMAGE = True
__C.TEST.WRITE_IMAGE_PATH = 'imgs/detection/'
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE = None
__C.TEST.SHOW_LABEL = True
__C.TEST.SCORE_THRESHOLD = 0.3
__C.TEST.IOU_THRESHOLD = 0.45
