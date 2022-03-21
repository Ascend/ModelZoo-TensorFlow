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
#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/8/26 16:07
# @Author  : XJTU-zzf
# @FileName: test_SR.py
"""
# > Script for evaluating 2x(4x, 8x) SISR models
#    - Paper: https://arxiv.org/pdf/1909.09437.pdf
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
import os
import time
import ntpath
import random

import datetime
import numpy as np
from scipy import misc
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import model_from_json
# local libs
from kutils.data_utils import getPaths, preprocess, deprocess
from PIL import Image
from kutils.print_config import print_config_test
# logger
from kutils import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # less logs

# 将将设置的超参数用tf.app.flag包装
flags = tf.flags
flags.DEFINE_string(name='test_mode', default='2x', help='choice=[2x, 4x, 8x]')
flags.DEFINE_string(name='chip', default='gpu', help='choice=[gpu, npu, cpu]')
flags.DEFINE_string(name="data_dir", default='/root/project/dataset/SRDRM/USR248/TEST/',
                    help='for testing arbitrary local data')
flags.DEFINE_string(name="obs_dir", default="obs://srdrm/", help="obs result path, not need on gpu and apulis platform")
flags.DEFINE_string(name='model_name', default="srdrm-gan", help="choice=[srdrm, srdrm-gan]")
flags.DEFINE_string(name='test_epoch', default='20', help='要测试的epoch')
flags.DEFINE_string(name='result', default='', help='使用NPU时需要用到，其余训练状态下为空')
flags.DEFINE_boolean(name="profiling", default=False, help="profiling for performance or not")
flags.DEFINE_string(name="platform", default="linux",
                    help="Run on linux/apulis/modelarts platform. Modelarts Platform has some extra data copy operations")
Flags = flags.FLAGS

if Flags.chip == 'gpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # set GPU:0
    # 设置set_session,与GPU有关
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # 设置GPU显存按需增长
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

elif Flags.chip == 'npu':
    from npu_bridge.npu_init import *

    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    sess = tf.Session(config=sess_config)
else:
    config = tf.compat.v1.ConfigProto()
    sess = tf.Session(config=config)

# random.seed(256)
# os.environ['PYTHONHASHSEED'] = str(256)
# np.random.seed(256)
# tf.compat.v1.set_random_seed(256)


K.set_session(sess)

if Flags.test_mode == '2x':
    lr_shape = (240, 320)
    dataset_name = 'USR_2x'
    scale = 2
elif Flags.test_mode == '4x':
    lr_shape = (120, 160)
    dataset_name = 'USR_4x'
    scale = 4
else:
    lr_shape = (60, 80)
    dataset_name = 'USR_8x'
    scale = 8

data_dir = os.path.join(Flags.data_dir, 'lr_{}'.format(Flags.test_mode))
test_paths = getPaths(data_dir)

# load specific model
if Flags.model_name == 'srdrm':
    ckpt_name = "model_{}_".format(Flags.test_epoch)
else:
    ckpt_name = "model_g_{}_".format(Flags.test_epoch)

checkpoint_dir = os.path.join(Flags.result, 'checkpoints', dataset_name, Flags.model_name)
model_h5 = os.path.join(checkpoint_dir, ckpt_name + ".h5")
model_json = os.path.join(checkpoint_dir, ckpt_name + ".json")
# sanity
assert (os.path.exists(model_h5) and os.path.exists(model_json)), '{},{}模型不存在'.format(model_h5, model_json)

# 记录控制台的输出的日志文件
logging_file = os.path.join(Flags.result, "task_log/test_{}_epoch_{}".format(Flags.chip, Flags.test_epoch), dataset_name,
                            Flags.model_name)
if not os.path.exists(logging_file):
    os.makedirs(logging_file)
log = Logger.Log(os.path.join(logging_file, "{}.log".format(str(datetime.datetime.now()).replace(":", "-").replace(".","-"))))

# load json and create model
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()

generator = model_from_json(loaded_model_json)

# load weights into new model
generator.load_weights(model_h5)

# create dir for output test data
prefix_dir = os.path.join(Flags.result, '{}_test_output'.format(Flags.chip),
                          'epoch_{}'.format(Flags.test_epoch), dataset_name)

samples_dir = os.path.join(prefix_dir, Flags.model_name)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

print_config_test(Flags, log)
log.info("========================TEST=========================")
log.info("| task name: Test dataset lr_{} using model {} ckpt name : {}".format(Flags.test_mode, Flags.model_name,
                                                                                ckpt_name))
log.info("| test_data_path: {}".format(data_dir))
log.info("| checkpoints dir: {}".format(checkpoint_dir))
log.info("| Logger file: {}".format(logging_file))
log.info("| {0} test images are loaded".format(len(test_paths)))
log.info("| output_dir: {}".format(samples_dir))
log.info("=====================================================")

# testing loop
times = []
s = time.time()
for img_path in test_paths:
    # prepare data
    img_name = ntpath.basename(img_path).split('.')[0]
    # img_lr_old = misc.imread(img_path, mode='RGB').astype(np.float64)
    # img_lr_old = misc.imresize(img_lr_old, lr_shape)

    img_lr = Image.open(img_path).convert('RGB')
    img_lr = np.array(img_lr.resize((lr_shape[1], lr_shape[0])), dtype=float)

    im = preprocess(img_lr)
    im = np.expand_dims(im, axis=0)
    # generate enhanced image
    s = time.time()
    gen = generator.predict(im)
    gen = deprocess(gen)  # Rescale to 0-1
    tot = time.time() - s
    times.append(tot)
    # save sample images
    Image.fromarray(np.uint8(gen[0]*255)).convert("RGB").save(os.path.join(samples_dir, img_name + '_gen.jpg'))
    # misc.imsave(os.path.join(samples_dir, img_name + '_gen.jpg'), gen[0])
    log.info("tested: {0}".format(img_path))

# some statistics
num_test = len(test_paths)
if num_test == 0:
    log.info("\nFound no images for test")
else:
    log.info("\nTotal images: {0}".format(num_test))
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
    log.info("Time taken: {0} sec at {1} fps".format(Ttime, 1. / Mtime))
    log.info("\nSaved generated images in in {0}\n".format(samples_dir))

    if Flags.platform.lower() == 'modelarts':
        from help_modelarts import modelarts_result2obs

        modelarts_result2obs(Flags)
sess.close()
