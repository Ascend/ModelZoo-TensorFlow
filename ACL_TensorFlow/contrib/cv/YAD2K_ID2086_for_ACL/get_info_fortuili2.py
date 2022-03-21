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
if not os.path.exists("evalue/accuracy"):
    os.mkdir("evalue/accuracy")
if not os.path.exists("evalue/accuracy/detections"):
    os.mkdir("evalue/accuracy/detections")
if not os.path.exists("evalue/accuracy/groundtruths"):
    os.mkdir("evalue/accuracy/groundtruths")


def read_data(data_path, input_shape=None):
    if data_path.endswith('.bin'):
        data = np.fromfile(data_path, dtype=np.float32)
        data = data.reshape(input_shape)
    elif data_path.endswith('.npy'):
        data = np.load(data_path)
    return data



yolo_outputs = yolo_head(model_body.output, anchors, len(classes))  # 对网络输出进行编码
input_image_shape = K.placeholder(shape=(2,))
boxes1, scores1, pred_classes = yolo_eval(yolo_outputs, input_image_shape, score_threshold=0.05,
                                        iou_threshold=0.6)

### 得到预测值
total = 0
current = 0
with open("./data_process/2007_test.txt", "r") as f:
    for tmp in f.readlines():
        total += 1
        test_path = tmp.split(" ")[0]
        img_name = test_path.split("/")[-1].split(".jpg")[0]
        sess = K.get_session()
        starttime = time.time()
        boxes, scores, cls = sess.run(
            [boxes1, scores1, pred_classes],
            feed_dict={
                model_body.output: read_data("out_bin/"+img_name + '_output_0.bin', [1, 13, 13, 125]),
                input_image_shape: [input_shape[1], input_shape[0]],
                K.learning_phase(): 0
            })
        endtime = time.time()
        current += float(endtime - starttime)
        print('Found {} boxes for {}'.format(len(boxes), test_path))
        with open("evalue/accuracy/detections/" + img_name + ".txt", "w") as f:
            for data in zip(cls, scores, boxes):
                f.write(" ".join([str(classes[data[0] ]), str(data[1] ),
                                  str(data[2][0] ), str(data[2][1] ),
                                  str(data[2][2] ), str(data[2][3] )]))
                f.write("\n")	
print("fps:", 1 / (current / total))




#得到真实值
all_files = []
with open("./data_process/2007_test.txt","r") as f:
    all_files = f.readlines()
    for file in all_files:
        image = cv2.imread(file.split(" ")[0])
        origin_shape = image.shape[0:2]
        x_scale, y_scale = float(input_shape[1]) / origin_shape[1], float(input_shape[0])/ origin_shape[0]
        filename= file.split(" ")[0].split("/")[-1].split(".jpg")[0]
        with open("evalue/accuracy/groundtruths/"+filename+".txt","w",encoding='utf_8') as ff:
            infos = file.split(" ")[1:]
            print(file)
            for info in infos:
                classname = classes[int(info.split(",")[-1].strip())]
                tmp = list(map(int,info.split(",")[:-1]))
                tmps =[str(tmp[0]*x_scale),str(tmp[1]*y_scale),str(tmp[2]*x_scale),str(tmp[3]*y_scale)]
                result = " ".join(tmps)
                ff.write(classname + " " + result+"\n")