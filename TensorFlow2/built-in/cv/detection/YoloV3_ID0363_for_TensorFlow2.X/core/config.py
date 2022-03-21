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
# ============================================================================

from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

# Set the class name
#__C.YOLO.CLASSES              = "./data/classes/coco.names"
__C.YOLO.CLASSES              = "./data/classes/yymnist.names"
__C.YOLO.ANCHORS              = "./data/anchors/basline_anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5

# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/dataset/yymnist_train.txt"
__C.TRAIN.BATCH_SIZE          = 4
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = [416]
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.EPOCHS              = 30



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/dataset/yymnist_test.txt"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 544
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.3
__C.TEST.IOU_THRESHOLD        = 0.45


