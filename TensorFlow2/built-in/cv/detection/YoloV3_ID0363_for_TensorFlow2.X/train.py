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

import os
import time
import shutil
import argparse
import pprint
import sys
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOv3, decode, compute_loss
from core.config import cfg

# npu modify begin
import npu_device
import ast

parser = argparse.ArgumentParser(description='Train classification network')

parser.add_argument('--dataset_dir',
                    help='dataset path',
                    default=None,
                    type=str)

parser.add_argument('--train_epochs',
                    help='the number of training epoch',
                    type=int,
                    default=1)

parser.add_argument('--batch_size',
                    help='batch size',
                    type=int,
                    default=8)

parser.add_argument('--ckpt_path',
                    help='the path of save model check point',
                    type=str,
                    default=None)

parser.add_argument('--data_dump_flag',
                    help='data dump flag, default is False',
                    type=str,
                    default='False')

parser.add_argument('--over_dump',
                    help='whether or not over detection, default is False',
                    type=str,
                    default='False')

parser.add_argument('--profiling',
                    help='whether profiling for performance debug, default is False',
                    type=str,
                    default='False')

parser.add_argument('--precision_mode',
                    help='allow_fp32_to_fp16/force_fp16/must_keep_origin_dtype/allow_mix_precision.',
                    type=str,
                    default='allow_fp32_to_fp16')

parser.add_argument('--data_dump_path',
                    help='the path of save dump data',
                    type=str,
                    default='/home/data')

parser.add_argument('--data_dump_step',
                    help='data dump step, default is 10',
                    type=str,
                    default='10')

parser.add_argument('--over_dump_path',
                    help='the path of save over dump data',
                    type=str,
                    default='/home/data')

parser.add_argument('--profiling_dump_path',
                    help='the path of save profiling data',
                    type=str,
                    default='/home/data')
parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval,
                    help='use_mixlist flag, default is False')
parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval,
                    help='fusion_off flag, default is False')
parser.add_argument('--mixlist_file', default="ops_info.json", type=str,
                    help='mixlist file name, default is ops_info.json')
parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,
                    help='fusion_off file name, default is fusion_switch.cfg')
args, unknown_args = parser.parse_known_args()


def npu_config(args):
    npu_config = {}

    if args.data_dump_flag == 'True':
        npu_device.global_options().dump_config.enable_dump = True
        npu_device.global_options().dump_config.dump_path = args.data_dump_path
        npu_device.global_options().dump_config.dump_step = args.data_dump_step
        npu_device.global_options().dump_config.dump_mode = "all"

    if args.over_dump == 'True':
        npu_device.global_options().dump_config.enable_dump_debug = True
        npu_device.global_options().dump_config.dump_path = args.over_dump_path
        npu_device.global_options().dump_config.dump_debug_mode = "all"

    if args.profiling == 'True':
        npu_device.global_options().profiling_config.enable_profiling = True
        profiling_options = '{"output":"' + args.profiling_dump_path + '", \
                            "training_trace":"on", \
                            "task_trace":"on", \
                            "aicpu":"on", \
                            "fp_point":"", \
                            "bp_point":""}'
        npu_device.global_options().profiling_config.profiling_options = profiling_options

    npu_device.global_options().precision_mode = args.precision_mode
    npu_device.global_options().variable_memory_max_size = str("4*1024*1024*1024")
    npu_device.global_options().graph_memory_max_size = str("27*1024*1024*1024")

    if args.use_mixlist and args.precision_mode == 'allow_mix_precision':
        npu_device.global_options().modify_mixlist = args.mixlist_file
    if args.fusion_off_flag:
        npu_device.global_options().fusion_switch_file = args.fusion_off_file
    npu_device.open().as_default()


npu_config(args)

cfg.YOLO.CLASSES = os.path.join(args.dataset_dir, "data/classes/yymnist.names")
cfg.YOLO.ANCHORS = os.path.join(args.dataset_dir, "data/anchors/basline_anchors.txt")
cfg.TRAIN.ANNOT_PATH = os.path.join(args.dataset_dir, "data/dataset/yymnist_train.txt")
cfg.TEST.ANNOT_PATH = os.path.join(args.dataset_dir, "data/dataset/yymnist_test.txt")
cfg.TEST.DECTECTED_IMAGE_PATH = os.path.join(args.dataset_dir, "data/detection")

trainset = Dataset('train', args.dataset_dir, args.batch_size)
logdir = "./data/log"
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = args.train_epochs * steps_per_epoch

input_tensor = tf.keras.layers.Input([416, 416, 3])
conv_tensors = YOLOv3(input_tensor)

output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor, i)
    output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)

model = tf.keras.Model(input_tensor, output_tensors)
optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)
'''
def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss=conf_loss=prob_loss=0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" %(global_steps, optimizer.lr.numpy(),
                                                          giou_loss, conf_loss,
                                                          prob_loss, total_loss))
        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps *cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()
'''


# npu modify begin
@tf.function
def train_execute(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        npu_device.distribute.all_reduce(gradients, "mean")
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return giou_loss, conf_loss, prob_loss, total_loss


def train_step(image_data, target):
    start_time = time.time()
    giou_loss, conf_loss, prob_loss, total_loss = train_execute(image_data, target)
    cost_time = time.time() - start_time
    tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
             "prob_loss: %4.2f   total_loss: %4.2f   perf: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                                     giou_loss, conf_loss,
                                                                     prob_loss, total_loss,
                                                                     cost_time))
    # update learning rate
    global_steps.assign_add(1)
    if global_steps < warmup_steps:
        lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
    else:
        lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
            (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
        )
    optimizer.lr.assign(lr.numpy())

    # writing summary data
    with writer.as_default():
        tf.summary.scalar("lr", optimizer.lr, step=global_steps)
        tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
        tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
        tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
        tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
    writer.flush()


# npu  modify end

for epoch in range(args.train_epochs):
    for image_data, target in trainset:
        train_step(image_data, target)
    # model.save_weights("./yolov3")
