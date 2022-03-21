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
import json
from PIL import Image, ImageDraw
import os
import cv2
import pandas as pd
from tqdm import tqdm
import shutil
from create_tfrecords import create
import random

IMAGES_DIR = './WIDER/WIDER_train/images/'
BOXES_PATH = './WIDER/wider_face_split/wider_face_train_bbx_gt.txt'
RESULT_DIR = './WIDER/TEST/'
num_shards=150


# collect paths to all images

all_paths = []
for path, subdirs, files in tqdm(os.walk(IMAGES_DIR)):
    for name in files:
        all_paths.append(os.path.join(path+'/', name))


metadata = pd.DataFrame(all_paths, columns=['full_path'])

# strip root folder
metadata['path'] = metadata.full_path.apply(lambda x: os.path.relpath(x, IMAGES_DIR).replace('\\','/'))


# see all unique endings
metadata.path.apply(lambda x: x.split('.')[-1]).unique()
print(len(metadata))

# read annotations
with open(BOXES_PATH, 'r') as f:
    content = f.readlines()
    content = [s.strip() for s in content]

# split annotations by image
boxes = {}
num_lines = len(content)
i = 0
name = None

while i < num_lines:
    s = content[i]
    # print(s)
    if s.endswith('.jpg'):
        if name is not None:
            if len(boxes[name]) != num_boxes:
                # print(num_boxes)
                boxes[name]=[]
        name = s
        boxes[name] = []
        i += 1
        # print(content[i])
        num_boxes = int(content[i])
        i += 1
    else:
        xmin, ymin, w, h = s.split(' ')[:4]
        xmin, ymin, w, h = int(xmin), int(ymin), int(w), int(h)
        if h <= 0 or w <= 0:
            num_boxes -= 1
        else:
            boxes[name].append((xmin, ymin, w, h))
        i += 1

def draw_boxes_on_image(path, boxes):

    image = Image.open(path)
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size

    for b in boxes:
        xmin, ymin, w, h = b
        xmax, ymax = xmin + w, ymin + h

        fill = (255, 255, 255, 45)
        outline = 'red'
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            fill=fill, outline=outline
        )
    return image

def get_annotation(path, width, height):
    name = path.split('/')[-1]
    annotation = {
        "filename": name,
        "size": {"depth": 3, "width": width, "height": height}
    }
    objects = []
    for b in boxes[path]:
        xmin, ymin, w, h = b
        xmax, ymax = xmin + w, ymin + h
        objects.append({
            "bndbox": {"ymin": ymin, "ymax": ymax, "xmax": xmax, "xmin": xmin},
            "name": "face"
        })
    annotation["object"] = objects
    return annotation


# create a folder for the converted dataset
shutil.rmtree(RESULT_DIR, ignore_errors=True)
os.mkdir(RESULT_DIR)
os.mkdir(os.path.join(RESULT_DIR, 'images'))
os.mkdir(os.path.join(RESULT_DIR, 'annotations'))

for T in tqdm(metadata.itertuples()):

    # get width and height of an image
    print()
    image = cv2.imread(T.full_path)
    #     image2 = Image.open(T.full_path)
    #     draw = ImageDraw.Draw(image2, 'RGBA')
    #     print(image2)
    h, w, c = image.shape
    assert c == 3

    # name of the image
    name = T.path.split('/')[-1]
    assert name.endswith('.jpg')

    # copy the image
    shutil.copy(T.full_path, os.path.join(RESULT_DIR, 'images', name))

    # save annotation for it
    d = get_annotation(T.path, w, h)
    json_name = name[:-4] + '.json'
    json.dump(d, open(os.path.join(RESULT_DIR, 'annotations', json_name), 'w'))



images_dir=RESULT_DIR+'images/'
annotations_dir=RESULT_DIR+'annotations/'
result_dir=RESULT_DIR+'output/'

create(images_dir,annotations_dir,result_dir,num_shards)