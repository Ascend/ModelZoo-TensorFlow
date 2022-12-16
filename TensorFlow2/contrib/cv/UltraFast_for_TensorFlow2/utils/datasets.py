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
# Copyright 2022 Huawei Technologies Co., Ltd
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


from unsupervised_llamas.label_scripts.spline_creator import get_horizontal_values_for_four_lanes
import tensorflow as tf
import numpy as np
import json
import os
import cv2
from scipy.interpolate import interp1d


from utils.grid import generate_grid


LLAMAS_SHAPE = (717, 1276)
DTYPE = tf.float32


def resize_img(img, shape):
    w = img.shape[1]
    h = img.shape[0]
    
    ratio = shape[1] / shape[0]
    
    tgt = img[h-int(w/ratio):h,0:w]
    tgt = cv2.resize(tgt, (shape[1], shape[0]))
    return tgt.astype(np.float32) / 255
    #return tgt


class LlamasProcessor:
    def __init__(self, cls_shape, image_shape):
        self.cls_shape = cls_shape
        self.image_shape = image_shape
        self.llamas_shape = LLAMAS_SHAPE
        
    def load_image(self, file):
        file = file.numpy().decode("utf-8")
        # print(file)
        with open(file, 'r') as fp:
            meta = json.load(fp)
        image_path = os.path.dirname(file)
        image_path = image_path.replace('labels', 'color_images')
        img_name = meta['image_name'] + '_color_rect.png'
        fname = os.path.join(image_path, img_name)
        # print(fname)
        img = cv2.imread(fname)
        
        return img
    
    def read_image(self, file):
        # print(file)
        img = self.load_image(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return resize_img(img, self.image_shape)

    def generate_grid_llamas(self, file):
        #img = self.load_image(file)
        file = file.numpy().decode("utf-8")
        lanes = get_horizontal_values_for_four_lanes(file)
        
        return generate_grid(lanes, self.cls_shape, self.llamas_shape, delete_lanes=(0,3))

    def process_json(self, json):
        img = tf.py_function(self.read_image, [json], Tout=DTYPE)
        grid = tf.py_function(self.generate_grid_llamas, [json], Tout=DTYPE)

        img = tf.reshape(img, shape=self.image_shape)
        grid = tf.reshape(grid, shape=self.cls_shape)
        return img, grid


class LabelmeProcessor:
    def __init__(self, cls_shape, image_shape):
        self.cls_shape = cls_shape
        self.image_shape = image_shape
        
    # Creates points for the left and right lane based on spline points
    # that are read from the json input file
    def get_points(self, file):
        x_points = [[], []]
        y_points = [[], []]

        with open(file, 'r') as fp:
            meta = json.load(fp)

        if meta['shapes'][0]['label'] == 'L':
            l = 0
            r = 1
        else:
            l = 1
            r = 0

        for x, y in meta['shapes'][l]['points']:
            x_points[l].append(x)
            y_points[l].append(y)

        for x, y in meta['shapes'][r]['points']:
            x_points[r].append(x)
            y_points[r].append(y)

        max_val = [0] * 2
        low_val = [0] * 2
        max_val[l] = np.max(y_points[l])
        low_val[l] = np.min(y_points[l])
        max_val[r] = np.max(y_points[r])
        low_val[r] = np.min(y_points[r])

        points = [[], []]
        f = [0] * 2
        f[l] = interp1d(y_points[l], x_points[l])
        f[r] = interp1d(y_points[r], x_points[r])

        for i in range(2):
            for j in range(meta['imageHeight']):
                if j < low_val[i] or j > max_val[i]:
                    points[i].append(-1)
                else:
                    points[i].append(f[i](j).item(0))
        return [points[l], points[r]]


    def read_image(self, file):   
        file = file.numpy().decode("utf-8")
        with open(file, 'r') as fp:
            meta = json.load(fp)

        image_path = os.path.dirname(file)
        img_name = meta['imagePath']
        fname = os.path.join(image_path, img_name)
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return resize_img(img, self.image_shape).astype(np.float32)
    
    def generate_grid_custom(self, file):
        file = file.numpy().decode("utf-8")
        with open(file, 'r') as fp:
            meta = json.load(fp)
        shape = (meta['imageHeight'], meta['imageWidth'])
        lines = self.get_points(file)
        
        return generate_grid(lines, self.cls_shape, shape)
    
    def process_json(self, json):
        img = tf.py_function(self.read_image, [json], Tout=DTYPE)
        grid = tf.py_function(self.generate_grid_custom, [json], Tout=DTYPE)

        img = tf.reshape(img, shape=self.image_shape)
        grid = tf.reshape(grid, shape=self.cls_shape)
        return img, grid


def load_json_dataset(json_file_pattern, processor, max_records=None, shuffle_size=100000, premap_func=None):
    ds = tf.data.Dataset.list_files(json_file_pattern)
    ds = ds.shuffle(shuffle_size)
    
    if max_records is not None:
        ds = ds.take(max_records)
        
    if premap_func is not None:
        ds = premap_func(ds)
        
    ds = ds.map(processor.process_json, num_parallel_calls=32)
    return ds

    
def llamas_dataset(json_file_pattern, cls_shape, image_shape, max_records=None, shuffle_size=100000, premap_func=None):
    processor = LlamasProcessor(cls_shape, image_shape)
    
    ds = load_json_dataset(json_file_pattern, processor, max_records, shuffle_size, premap_func)
    
    return ds


def labelme_dataset(json_file_pattern, cls_shape, image_shape, max_records=None, shuffle_size=100000, premap_func=None):
    processor = LabelmeProcessor(cls_shape, image_shape)
    
    ds = load_json_dataset(json_file_pattern, processor, max_records, shuffle_size, premap_func)
    
    return ds
