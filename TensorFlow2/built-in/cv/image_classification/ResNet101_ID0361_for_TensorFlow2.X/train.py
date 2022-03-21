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

from __future__ import absolute_import, division, print_function
import argparse
import os
import pprint
import shutil
import sys

import tensorflow as tf
from models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
import config
from prepare_data import generate_datasets
import math
import time
import npu_device
import ast

def parse_args():
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

    return args


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
    if args.use_mixlist and args.precision_mode == 'allow_mix_precision':
        npu_device.global_options().modify_mixlist = args.mixlist_file
    if args.fusion_off_flag:
        npu_device.global_options().fusion_switch_file = args.fusion_off_file
    npu_device.open().as_default()


def get_model():
    model = resnet_50()
    if config.model == "resnet18":
        model = resnet_18()
    if config.model == "resnet34":
        model = resnet_34()
    if config.model == "resnet101":
        model = resnet_101()
    if config.model == "resnet152":
        model = resnet_152()
    model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    model.summary()
    return model


if __name__ == '__main__':
    args = parse_args()
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    npu_config(args)
    # get the original_dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets(args)

    # create model
    model = get_model()

    # define loss and optimizer
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adadelta()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')


    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)


    @tf.function
    def valid_step(images, labels):
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)


    # start training
    for epoch in range(args.train_epochs):
        # train_loss.reset_states()
        # train_accuracy.reset_states()
        # valid_loss.reset_states()
        # valid_accuracy.reset_states()
        train_acc_ave = 0
        train_acc_total = 0

        step = 0
        cost_time = 0
        for images, labels in train_dataset:
            start_time = time.time()
            step += 1
            train_step(images, labels)
            train_acc_total += train_accuracy.result()

            cost_time += (time.time() - start_time)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}, perf: {:.5f}".format(epoch + 1,
                                                                                                   args.train_epochs,
                                                                                                   step,
                                                                                                   math.ceil(
                                                                                                       train_count / args.batch_size),
                                                                                                   train_loss.result(),
                                                                                                   train_accuracy.result(),
                                                                                                   cost_time))
            cost_time = 0
        train_acc_ave = train_acc_total / (math.ceil(train_count / args.batch_size))
        test_acc_ave = 0
        test_acc_total = 0
        test_count = 0

        for valid_images, valid_labels in valid_dataset:
            valid_step(valid_images, valid_labels)
            test_acc_total += valid_accuracy.result()
            test_count += 1

        test_acc_ave = test_acc_total / test_count

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  args.train_epochs,
                                                                  train_loss.result(),
                                                                  train_acc_ave,
                                                                  valid_loss.result(),
                                                                  test_acc_ave))

    model.save_weights(filepath=config.save_model_dir, save_format='tf')
