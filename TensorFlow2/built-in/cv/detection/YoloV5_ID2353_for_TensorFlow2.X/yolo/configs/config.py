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
import argparse
import ast

parser = argparse.ArgumentParser()

parser.add_argument('--train_annotations_dir', type=str, default='../data/voc2012/VOCdevkit/VOC2012/train.txt', help='train annotations path')
parser.add_argument('--test_annotations_dir', type=str, default='../data/voc2012/VOCdevkit/VOC2012/valid.txt', help='test annotations path')
parser.add_argument('--class_name_dir', type=str, default='../data/voc2012/VOCdevkit/VOC2012/voc2012.names', help='classes name path')
parser.add_argument('--yaml_dir', type=str, default='configs/yolo-m-mish.yaml', help='model.yaml path')
parser.add_argument('--log_dir', type=str, default='../logs', help='log path')
parser.add_argument('--checkpoint_dir', type=str, default='../weights', help='saved checkpoint path')
parser.add_argument('--saved_model_dir', type=str, default='../weights/yolov5', help='saved pb model path')

parser.add_argument('--n_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=4, help='total batch size for all GPUs')
parser.add_argument('--multi_gpus', type=bool, default=False)
parser.add_argument('--init_learning_rate', type=float, default=3e-4)
parser.add_argument('--warmup_learning_rate', type=float, default=1e-6)
parser.add_argument('--warmup_epochs', type=int, default=2)
parser.add_argument('--img_size', type=int, default=640, help='image target size')
parser.add_argument('--mosaic_data', type=bool, default=False, help='if mosaic data')
parser.add_argument('--augment_data', type=bool, default=True, help='if augment data')
parser.add_argument('--anchor_assign_method', type=str, default='wh', help='assign anchor by wh or iou')
parser.add_argument('--anchor_positive_augment', type=bool, default=True, help='extend the neighbour to positive')
parser.add_argument('--label_smoothing', type=float, default=0.02, help='classification label smoothing')
parser.add_argument("--export_model", action='store_true', help='if save pb model')
parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='the path to save over dump data')
parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval,
                    help='if or not over detection, default is False')
parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval,
                    help='data dump flag, default is False')
parser.add_argument('--data_dump_step', default="10",
                    help='data dump step, default is 10')
parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval,
                    help='use_mixlist flag, default is False')
parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval,
                    help='fusion_off flag, default is False')
parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
args = parser.parse_args()
params = vars(args)
