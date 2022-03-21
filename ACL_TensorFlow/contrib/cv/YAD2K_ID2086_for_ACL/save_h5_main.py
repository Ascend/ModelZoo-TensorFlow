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
import time
import cv2
import numpy as np
from data_process.config import input_shape
import tensorflow as tf
import numpy as np
import keras.backend as K
from detect import yolo_eval
from nets.yololoss import yolo_head
from data_process.config import anchors,classes,input_shape,batch_size
from nets.v2net import model_body
import cv2
from PIL import Image

CUDA_VISIBLE_DEVICES = 3


model_path = r"ep100-loss1.568-val_loss18.650.h5"
model_body.load_weights(model_path) #加载权重
model_body.summary()
print("正在生成中")
model_body.save("./log/new_2.h5")
print("正在生成结束")
