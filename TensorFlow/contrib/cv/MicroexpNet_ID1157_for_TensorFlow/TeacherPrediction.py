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

import os
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import  layers
from tensorflow.keras import Model
import json
from npu_bridge.npu_init import *
import  sys
from  tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

def define_mode(InceptionV3_weight_path,weights_path):
    # InceptionV3_weight_path = '/cache/dataset/TeacherExpNet_CK.h5'
    # pre_mode = InceptionV3(include_top=False,weights="/cache/dataset/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",input_shape=(256,256,3))
    pre_mode = InceptionV3(include_top=False,weights=weights_path,input_shape=(256,256,3))
    last_layer=pre_mode.get_layer('mixed10')
    last_output=last_layer.output
    x=layers.GlobalAvgPool2D()(last_output)
    x=layers.Dense(8,activation='softmax')(x)
    mode = Model(inputs=pre_mode.input,outputs=x)
    mode.load_weights(InceptionV3_weight_path)
    return mode

def teacherPrediction(datapath,outputPredictionPos,InceptionV3_weight_path,weights_path):
    mode = define_mode(InceptionV3_weight_path,weights_path)
    print("start!")
    datalist=os.listdir(datapath)
    f=open(outputPredictionPos+"/preds.json","w")
    preds = {}
    for i in range(len(datalist)):
        img=cv2.imread(datapath+"/"+datalist[i])
        #img=cv2.resize(img,(299,299))
        gray_im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_im = cv2.cvtColor(gray_im, cv2.COLOR_GRAY2RGB)
        gray_im = preprocess_input(gray_im)
        gray_im = np.array(gray_im).reshape(-1,256,256,3)
        pred=mode.predict(gray_im)
        preds[datapath + "/" + datalist[i]]=pred.tolist()
    js=json.dumps(preds)
    f.write(js)
    f.close()
    print("finish")