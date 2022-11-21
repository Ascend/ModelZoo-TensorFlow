# -*- coding: utf-8 -*-
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
"""
Created on Sun Feb 24 22:32:37 2019

@author: wmy
"""
#***** NPU modify begin*****
from npu_bridge.npu_init import *
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras import backend as K
#***** NPU modify end*****

import scipy
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from model import Pix2Pix
from PIL import Image
import argparse
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)

def predict_single_image(pix2pix, image_path, save_path, weights_path):
    pix2pix.generator.load_weights(weights_path + '/generator_weights.h5')
    image_B = imread(image_path)
    image_B = scipy.misc.imresize(image_B, (pix2pix.nW, pix2pix.nH))
    images_B = []
    images_B.append(image_B)
    images_B = np.array(images_B)/127.5 - 1.
    generates_A = pix2pix.generator.predict(images_B)
    generate_A = generates_A[0]
    generate_A = np.uint8((np.array(generate_A) * 0.5 + 0.5) * 255)
    generate_A = Image.fromarray(generate_A)
    generated_image = Image.new('RGB', (pix2pix.nW, pix2pix.nH))
    generated_image.paste(generate_A, (0, 0, pix2pix.nW, pix2pix.nH))
    generated_image.save(save_path, quality=95)
    pass

def convert_to_gray_single_image(image_path, save_path, resize_height=256, resize_weidth=256): 
    img = Image.open(image_path)
    img_color = img.resize((resize_weidth, resize_height), Image.ANTIALIAS)
    img_gray = img_color.convert('L')
    img_gray = img_gray.convert('RGB')
    img_gray.save(save_path, quality=95)

#*****path to save weight****
flags.DEFINE_integer("epochs", 1200, "train epochs")
flags.DEFINE_string("new_weights_path", "./test/output/new_weights", "new weights file path")
flags.DEFINE_integer("batch_size", 4, "train batchsize")
flags.DEFINE_string("precision_mode", "allow_fp32_to_fp16", "train precision mode")
flags.DEFINE_integer("sample_interval", 10, "sample_interval")
flags.DEFINE_boolean("load_pretrained", False, "load_pretrained")

#***** NPU modify begin*****
sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
# custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(FLAGS.precision_mode)
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

# 动态输入
# custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
# custom_op.parameter_map["dynamic_input"].b = True
# custom_op.parameter_map["use_off_line"].b = True

sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
#sess = tf.Session(config=sess_config)
#K.set_session(sess)
#***** NPU modify end*****

gan = Pix2Pix()
print("weights_path:{}，epochs：{}，batchsize：{}".format(FLAGS.new_weights_path,FLAGS.epochs,FLAGS.batch_size))
with tf.Session(config=sess_config) as sess:
    K.set_session(sess)
    gan.train(weights_path=FLAGS.new_weights_path, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, sample_interval=FLAGS.sample_interval, load_pretrained=FLAGS.load_pretrained)
    predict_single_image(gan, './images/test_1.jpg', './images/generate_test_1.jpg', FLAGS.new_weights_path)

#gan.train(epochs=1200, batch_size=4, sample_interval=10, load_pretrained=True)
#gan.train(weights_path=FLAGS.weights_path, epochs=5, batch_size=3, sample_interval=10, load_pretrained=True)
#print("weights_path:{}，epochs：{}，batchsize：{}".format(FLAGS.weights_path,FLAGS.epochs,FLAGS.batch_size))
#gan.train(weights_path=FLAGS.weights_path, epochs=int(FLAGS.epochs), batch_size=int(FLAGS.batch_size), sample_interval=10, load_pretrained=True)

# predict_single_image(gan, './images/test_1.jpg', './images/generate_test_1.jpg')
# sess.close()
