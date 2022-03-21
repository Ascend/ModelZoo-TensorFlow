#!/usr/bin/env python3
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
# -*- coding: UTF-8 -*-
"""
Run this code to build and save tensorflow model with corresponding weight values for VNect
"""
from npu_bridge.npu_init import *


import os
import tensorflow as tf
from src.caffe2pkl import caffe2pkl
from src.vnect_model import VNect


def init_tf_weights(pfile, spath, model):
    # configurations
    PARAMSFILE = pfile
    SAVERPATH = spath

    if not tf.gfile.Exists(SAVERPATH):
        tf.gfile.MakeDirs(SAVERPATH)

    #npu modify begin
    #with tf.Session() as sess:
    with tf.Session(config=npu_config_proto()) as sess:
    #npu modify end
        saver = tf.train.Saver()
        model.load_weights(sess, PARAMSFILE)
        saver.save(sess, os.path.join(SAVERPATH, 'vnect_tf'))


# caffe model basepath
caffe_bpath = './models/caffe_model'
# caffe model files
prototxt_name = 'vnect_net.prototxt'
caffemodel_name = 'vnect_model.caffemodel'
# pickle file name
pkl_name = 'params.pkl'
pkl_file = os.path.join(caffe_bpath, pkl_name)
# tensorflow model path
tf_save_path = './models/tf_model'

if not os.path.exists(pkl_file):
    caffe2pkl(caffe_bpath, prototxt_name, caffemodel_name, pkl_name)

model = VNect()
init_tf_weights(pkl_file, tf_save_path, model)
