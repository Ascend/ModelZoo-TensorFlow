#!/usr/bin/python3
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
# -*- coding: utf-8 -*-
# @Time    : 2021/8/26 11:14
# @Author  : XJTU-zzf
# @FileName: train_genarative_models.py
"""
# > Script for training 2x(4x,8x) generative SISR models on USR-248 data
#    - Paper: https://arxiv.org/pdf/1909.09437.pdf
"""
from __future__ import print_function, division
import os
import datetime
import random

import numpy as np
# tf keras libs
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
import tensorflow as tf
from nets.gen_models import SRDRM_gen
# local libs
from kutils.plot_utils import save_val_samples
from kutils.data_utils import dataLoaderUSR, deprocess
from kutils.loss_utils import perceptual_distance, total_gen_loss
from kutils.print_config import print_config_train
# 导入记录日志的文件
from kutils import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # less logs

# 将设置的超参数用tf.app.flag包装
flags = tf.flags
flags.DEFINE_string(name='train_mode', default='8x', help='choice=[2x, 4x, 8x]')
flags.DEFINE_string(name='chip', default='gpu', help='choice=[gpu, npu, cpu]')
# training parameters
flags.DEFINE_integer(name='num_epochs', default=20, help='训练的epoch数')
flags.DEFINE_integer(name='batch_size', default=4, help='batch size')
flags.DEFINE_integer(name='sample_interval', default=500, help='per step')
flags.DEFINE_integer(name='ckpt_interval', default=4, help='多少个epoch保存训练的模型')
flags.DEFINE_string(name="obs_dir", default="obs://srdrm/", help="obs result path, not need on gpu and apulis platform")
flags.DEFINE_string(name="platform", default="linux",
                    help="Run on linux/apulis/modelarts platform. Modelarts Platform has some extra data copy "
                         "operations")
flags.DEFINE_string(name='output', default='', help='输出路径，只在在modelarts上训练时要用')
flags.DEFINE_boolean(name="profiling", default=False, help="profiling for performance or not")
flags.DEFINE_string(name="result", default="./result",
                    help="The result directory where the model checkpoints will be written.")
flags.DEFINE_string(name="data_path", default='/root/project/dataset/SRDRM/USR248/', help='数据集的存放路径')
flags.DEFINE_string(name='model_name', default='srdrm', help='model name')
flags.DEFINE_integer(name='start_epoch', default=0, help='start_epoch')
Flags = flags.FLAGS

if Flags.chip == 'gpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set GPU:0
    # 设置set_session,与GPU有关
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # 设置GPU显存按需增长
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

elif Flags.chip == 'npu':
    from npu_bridge.npu_init import *

    # os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = "1"
    sess_config = tf.compat.v1.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes('allow_mix_precision')
    # 设置自动调优
    # custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
    if Flags.profiling:
        custom_op.parameter_map["profiling_mode"].b = True
        custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
            '{"output":"/home/HwHiAiUser/output","task_trace":"on"}')
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    sess = tf.compat.v1.Session(config=sess_config)
else:
    config = tf.compat.v1.ConfigProto()
    sess = tf.Session(config=config)

# 固定随机种子，保证结果可复现
# random.seed(256)
# os.environ['PYTHONHASHSEED'] = str(256)
# np.random.seed(256)
# tf.set_random_seed(256)

K.set_session(sess)

# dataset and image information
channels = 3
hr_width, hr_height = 640, 480  # high res
"""
   根据训练模式来设定参数:
   lr_width, lr_height 使用的低分辨率的图像的宽和高
   dataset_name 使用数据集的名称，方便后期保存模型文件
   scale 相比于高分辨率图像的缩放尺度
"""
if Flags.train_mode == '2x':
    lr_width, lr_height = 320, 240
    dataset_name = 'USR_2x'
    scale = 2
elif Flags.train_mode == '4x':
    lr_width, lr_height = 160, 120
    dataset_name = 'USR_4x'
    scale = 4
else:
    lr_width, lr_height = 80, 60
    dataset_name = 'USR_8x'
    scale = 8

# input and output data
lr_shape = (lr_height, lr_width, channels)
hr_shape = (hr_height, hr_width, channels)
# load the data
data_loader = dataLoaderUSR(DATA_PATH=Flags.data_path, SCALE=scale)

# training parameters
steps_per_epoch = (data_loader.num_train // Flags.batch_size)
print('===steps_per_epoch===',steps_per_epoch)
num_step = Flags.num_epochs * steps_per_epoch

model_loader = SRDRM_gen(lr_shape, hr_shape, SCALE=scale)
# initialize the model
model = model_loader.create_model()

# print (model.summary())
# checkpoint directory
checkpoint_dir = os.path.join(Flags.output, "checkpoints/", dataset_name, Flags.model_name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# log directory
logging_file = os.path.join(Flags.output, "task_log/train_{}".format(Flags.chip), dataset_name, Flags.model_name)
if not os.path.exists(logging_file):
    os.makedirs(logging_file)
log = Logger.Log(os.path.join(logging_file,
                              "{}.log".format(str(datetime.datetime.now()).replace(":", "-").replace(".", "-"))))

# sample directory
samples_dir = os.path.join(Flags.output, "images", dataset_name, Flags.model_name)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

# compile model
# 单卡学习率改为0.0001
optimizer_ = Adam(0.0002, 0.5)
model.compile(optimizer=optimizer_, loss=total_gen_loss)

if Flags.start_epoch > 0:
    model_h5 = os.path.join(checkpoint_dir, ("model_%d" % Flags.start_epoch) + "_.h5")
    assert os.path.exists(model_h5), "模型文件不存在"
    model.load_weights(model_h5)

# training pipeline
step, epoch = Flags.start_epoch * steps_per_epoch, Flags.start_epoch

if Flags.start_epoch > 0:
    basic_step = step
else:
    basic_step = 0

# tensorboard writer
log_dir = os.path.join(Flags.output, "tensorboard_logs/{}/{}".format(dataset_name, Flags.model_name))
writer = tf.python.summary.FileWriter(log_dir)

print_config_train(Flags, log)
log.info("=======================TRAIN===========================")
log.info("| Using model: SRDRM for {}.".format(Flags.train_mode))
log.info("| Training: {0} with {1} data".format(Flags.model_name, dataset_name))
log.info("| Training start at epoch: {} .".format(Flags.start_epoch))
log.info("| num_step: {}".format(num_step))
log.info("| checkpoint path {}:".format(checkpoint_dir))
log.info("| Logger file: {}".format(logging_file))
log.info("| sample path:{}".format(samples_dir))
log.info("| tensorboard log dir: {}".format(log_dir))
log.info("======================================================")

train_time = []
# training pipeline
# accumulate frame processing times (without bootstrap)
start_step = step + 1 - basic_step
while step < num_step - 1:
    for i, (imgs_lr, imgs_hr) in enumerate(data_loader.load_batch(Flags.batch_size)):

        start_time = datetime.datetime.now()
        # train_gpu the generator
        loss_i = model.train_on_batch(imgs_lr, imgs_hr)
        # increment step, and show the progress

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        train_time.append(elapsed_time)
        step += 1

        if step % 50 == 0:
            end_step = step - basic_step
            Ttime = np.sum(train_time[start_step:end_step]) / (end_step - start_step)
            Mtime = np.mean(train_time[start_step:end_step]) / Flags.batch_size
            start_step = end_step
            log.info("[Step %5d/%5d] [Epoch %2d: batch %3d/%3d] [loss_i: %.6f] [%.4f seconds/step, %.4f fps]"
                     % (step, num_step, epoch, i + 1, steps_per_epoch, loss_i, Ttime, 1. / Mtime))
            summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='{}_g_loss'.format(dataset_name),
                                                                             simple_value=loss_i), ])
            writer.add_summary(summary, step)
        # validate and save generated samples at regular intervals
        if step % Flags.sample_interval == 0:
            imgs_lr, imgs_hr = data_loader.load_val_data(batch_size=Flags.batch_size)
            fake_hr = model.predict(imgs_lr)
            # 因为此处NPU不支持动态shape，所以将batchsize2改成两个batch 1
            # imgs_lr_f, imgs_hr_f = data_loader.load_val_data(batch_size=1)
            # fake_hr_f = model.predict(imgs_lr_f)
            # imgs_lr_s, imgs_hr_s = data_loader.load_val_data(batch_size=1)
            # fake_hr_s = model.predict(imgs_lr_s)
            # imgs_hr = np.concatenate([imgs_hr_f, imgs_hr_s])
            # fake_hr = np.concatenate([fake_hr_f, fake_hr_s])
            gen_imgs = np.concatenate([deprocess(fake_hr), deprocess(imgs_hr)])
            save_val_samples(samples_dir, gen_imgs, step)

    # increment epoch, save model at regular intervals
    epoch += 1
    # save model and weights
    if epoch % Flags.ckpt_interval == 0 or epoch == Flags.num_epochs:
        ckpt_name = os.path.join(checkpoint_dir, ("model_%d" % epoch))
        with open(ckpt_name + "_.json", "w") as json_file:
            json_file.write(model.to_json())
        model.save_weights(ckpt_name + "_.h5")
        log.info("Saved trained model in {0}".format(checkpoint_dir))

        if Flags.platform.lower() == 'modelarts':
            from help_modelarts import modelarts_result2obs

            modelarts_result2obs(Flags)

log.info("Total train_gpu time : %.4f s" % (np.sum(train_time[1:])))
log.info("Figure per seconds: %.4f " % ((1. / np.mean(train_time[1:])) * Flags.batch_size))
sess.close()
