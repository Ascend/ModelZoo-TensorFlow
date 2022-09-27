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
import numpy as np
from tqdm import tqdm
from PIL import Image
import xml.etree.ElementTree as ET
from utils_map import get_map

image_ids = open("/root/FasterRCNN/VOCdevkit/VOC2012/ImageSets/Main/val.txt").read().strip().split()

class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

def cvtColor(img):
    if len(np.shape(img)) == 3 and np.shape(img)[2] == 3:
        return img
    else:
        img = img.convert('RGB')
        return img

for image_id in tqdm(image_ids):
    with open('/root/FasterRCNN/FasterRCNN-PR/map_out/ground-truth/' + image_id + ".txt", "w") as new_f:
        root = ET.parse("/root/FasterRCNN/VOCdevkit/VOC2012/Annotations/" + image_id + ".xml").getroot()
        for obj in root.findall('object'):
            difficult_flag = False
            if obj.find('difficult') != None:
                difficult = obj.find('difficult').text
                if int(difficult) == 1:
                    difficult_flag = True
            obj_name = obj.find('name').text
            if obj_name not in class_names:
                continue
            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text

            if difficult_flag:
                new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
            else:
                new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

get_map(0.5, True)
