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
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extract ResNet feature
Author: Kaihua Tang
"""
#npu modify begin
from npu_bridge.npu_init import *
#npu modify end
import math
import time
import tensorflow as tf
import ResNet as resnet
import numpy as np
import scipy.io as scio
from scipy import misc
from utils import *

# image size
WIDTH = 224
HEIGHT = 224
CHANNELS = 3
#Number of output labels
LABELSNUM = 1200
#"Path of Label.npy"
label_path = "./label/label.npy"
#"Path of image file names"
image_name_path = "./label/name.npy"
# image path
parentPath = "./CACD2000_Crop/"


def CalculateFeature():
    """
    EXtract ResNet Feature by trained model
    model_path: The model we use
    feature_path: The path to save feature
    """
    model_path = "./model/03.npy"
    feature_path = "./resnet_feature.mat"

    #Lists that store name of image and its label
    testNameList = np.load(image_name_path)
    testLabelList = np.load(label_path)

    #num of total training image
    num_test_image = testLabelList.shape[0]
    #load all image data
    allImageData = load_all_image(testNameList, HEIGHT, WIDTH, CHANNELS, parentPath)
    #container for ResNet Feature
    res_feature = np.zeros((num_test_image, 2048))

    #npu modify begin
    #with tf.Session() as sess:
    with tf.Session(config=npu_config_proto()) as sess:
    #npu modify end
        images = tf.placeholder(tf.float32, shape = [None, WIDTH, HEIGHT, CHANNELS])

        # build resnet model
        resnet_model = resnet.ResNet(ResNet_npy_path = model_path)
        resnet_model.build(images, LABELSNUM, "softmax")

        sess.run(tf.global_variables_initializer())
        resnet_model.set_is_training(False)

        for i in range(num_test_image):
           if(i%1000 == 0):
                print(i)
           (minibatch_X, minibatch_Y) = get_minibatch([i], testLabelList, HEIGHT, WIDTH, CHANNELS, LABELSNUM, allImageData, True)
           pool2 = sess.run([resnet_model.pool2], feed_dict={images: minibatch_X})
           res_feature[i][:] = pool2[0][:]

        scio.savemat(feature_path,{'feature' : res_feature})

if __name__ == '__main__':
    CalculateFeature()
