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
import numpy as np
import tensorflow as tf
import os
import shutil
import tensorflow as tf
import PIL.Image
import numpy as np
from tqdm import tqdm

faceList_path=r"../FDDBFile/faceList.txt"
# output_dir=r"../outputs/"
output_dir=r"C:/Users/DELL/Desktop/huawei/New folder1/202284_19_57_23_328041"
output_file=r"../FDDBFile/detections.txt"

ofiles=os.listdir(output_dir)

pic_name=[]
with open(faceList_path) as f:
    temp=f.readlines()
    for i in temp:
        pic_name.append(i.strip())

predictions = []

for example in tqdm(pic_name):

    out0 = output_dir+"/"+example.replace("/","-")+".jpg_output_0.txt"
    out1 = output_dir+"/"+example.replace("/","-")+".jpg_output_1.txt"
    out2 = output_dir+"/"+example.replace("/","-")+".jpg_output_2.txt"
    # boxes, scores, num_boxes

    with open(out2) as f:
        temp=f.readlines()
        data=temp[0]
        data = data.strip("\n")  # 去除末尾的换行符
        data = data.strip("\t")  # 去除末尾的换行符
        data_split = data.split(" ")
        num_boxes=int(data_split[0])

    with open(out1) as f:
        temp=f.readlines()
        data=temp[0]
        data = data.strip("\n")  # 去除末尾的换行符
        data = data.strip("\t")  # 去除末尾的换行符
        data_split = data.split(" ")
        data_split=data_split[:num_boxes]
        scores = [float(x) for x in data_split]
        scores=np.expand_dims(np.array(scores),axis=1)

    with open(out0) as f:
        temp=f.readlines()
        data=temp[0]
        data = data.strip("\n")  # 去除末尾的换行符
        data = data.strip("\t")  # 去除末尾的换行符
        data_split = data.split(" ")
        data_split=data_split[:num_boxes*4]
        boxes = [float(x) for x in data_split]
        boxes=np.array(boxes).reshape((-1,4))

    # dataout=np.concatenate([boxes,scores],axis=1)
        ###recorver to raw image
    scaler = np.array([1024.0,
                       1024.0,
                       1024.0,
                       1024.0], dtype='float32')
    boxes = boxes * scaler

    scores=np.expand_dims(scores, 0).reshape([-1,1])

    #####the tf.nms produce ymin,xmin,ymax,xmax,  swap it in to xmin,ymin,xmax,ymax
    for i in range(boxes.shape[0]):
        boxes[i] = np.array([boxes[i][1], boxes[i][0], boxes[i][3],boxes[i][2]])
    fboxes=np.concatenate([boxes, scores],axis=1)
    predictions.append((example, fboxes[:,0:4], fboxes[:,4]))


with open(output_file, 'w') as f:
    for n, boxes, scores in predictions:
        f.write(n + '\n')
        f.write(str(len(boxes)) + '\n')
        for b, s in zip(boxes, scores):
            xmin, ymin, xmax, ymax = b
            h, w = int(ymax - ymin+1), int(xmax - xmin+1)
            f.write('{0} {1} {2} {3} {4:.4f}\n'.format(int(xmin), int(ymin), w, h, s))

