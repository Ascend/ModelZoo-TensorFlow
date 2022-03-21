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
"""
Train and evaluate SESEMI architecture for supervised learning augmented
with self-supervised task of recognizing geometric transformations
defined as 90-degree rotations with horizontal and vertical flips.
"""

# Python package imports
import os
import argparse
import struct
import numpy
import pickle
import glob
import time
import sys
import tensorflow as tf
import numpy as np
# SESEMI package imports
from networks import convnet, wrn, nin
from datasets import svhn, cifar10, cifar100
from utils import global_contrast_normalize, zca_whitener
from utils import stratified_sample, gaussian_noise, datagen, jitter
from utils import LRScheduler, DenseEvaluator, compile_sesemi
# Keras package imports
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


def open_sesemi():
    """
    open_sesemi
    """
    sess = tf.InteractiveSession()
    tf.global_variables_initializer()


    # Experiment- and dataset-dependent parameters.
    zca = True

    # Prepare the dataset.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_test = global_contrast_normalize(x_test)
    x_train = global_contrast_normalize(x_train)

    if zca:
        zca_whiten = zca_whitener(x_train)
        x_test = zca_whiten(x_test)

    x_test = x_test.reshape((len(x_test), 32, 32, 3))
    x_test = x_test.astype(np.float32)


    x_val = x_test
    data = []
    for x in x_val:
        t = jitter(x)
        noisy_x = gaussian_noise(x)
        noisy_t = gaussian_noise(t)
        flipx = np.fliplr(x)
        flipt = np.fliplr(t)
        noisy_flipx = gaussian_noise(flipx)
        noisy_flipt = gaussian_noise(flipt)
        data.append([x, t, noisy_x, noisy_t, flipx, flipt, noisy_flipx, noisy_flipt])
    data = np.vstack(data)

    input_x = data
    input_x = input_x.astype(np.float32)
    input_x_1 = input_x[0:10000, :, :, :]
    input_x_2 = input_x[10000:20000, :, :, :]
    input_x_3 = input_x[20000:30000, :, :, :]
    input_x_4 = input_x[30000:40000, :, :, :]
    input_x_5 = input_x[40000:50000, :, :, :]
    input_x_6 = input_x[50000:60000, :, :, :]
    input_x_7 = input_x[60000:70000, :, :, :]
    input_x_8 = input_x[70000:80000, :, :, :]
    input_x_1.tofile("bin/input_x_1.bin")
    input_x_2.tofile("bin/input_x_2.bin")
    input_x_3.tofile("bin/input_x_3.bin")
    input_x_4.tofile("bin/input_x_4.bin")
    input_x_5.tofile("bin/input_x_5.bin")
    input_x_6.tofile("bin/input_x_6.bin")
    input_x_7.tofile("bin/input_x_7.bin")
    input_x_8.tofile("bin/input_x_8.bin")
    y_test.tofile("bin/y_test.bin")

if __name__ == '__main__':
    open_sesemi()
