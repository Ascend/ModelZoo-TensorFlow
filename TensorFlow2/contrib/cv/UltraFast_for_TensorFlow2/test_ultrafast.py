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


#import npu_device as npu
#npu.open().as_default()

import time
from unsupervised_llamas.label_scripts.spline_creator import get_horizontal_values_for_four_lanes
import tensorflow as tf
import numpy as np
import json
import os
import cv2
import matplotlib.pyplot as plt
from utils import losses, metrics

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.savefig('result.png')

BASE_PATH = './data/llamas/labels/'
PARALLEL_JOBS = 4

NUM_LANES = 2
INPUT_SHAPE = (288, 800, 3)
CLS_SHAPE = (100, 20, NUM_LANES)
IMAGE_SHAPE = (288, 800, 3)
SHUFFLE_SIZE = 100000
TAKE_LLAMAS_MAX = 1024

LLAMAS_SHAPE = (717, 1276)
DTYPE = tf.float32


def resize_img(img, shape):
    w = img.shape[1]
    h = img.shape[0]

    ratio = shape[1] / shape[0]

    tgt = img[h - int(w / ratio):h, 0:w]
    tgt = cv2.resize(tgt, (shape[1], shape[0]))
    return tgt.astype(np.float32) / 255
    # return tgt


class LlamasDS:
    def __init__(self, cls_shape, image_shape):
        self.cls_shape = cls_shape
        self.image_shape = image_shape

    def load_image(self, file):
        file = file.numpy().decode("utf-8")

        with open(file, 'r') as fp:
            meta = json.load(fp)
        image_path = os.path.dirname(file)
        image_path = image_path.replace('labels', 'color_images')
        img_name = meta['image_name'] + '_color_rect.png'
        fname = os.path.join(image_path, img_name)
        img = cv2.imread(fname)

        return img

    def read_image(self, file):
        img = self.load_image(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return resize_img(img, self.image_shape)

    def generate_grid_llamas(self, file):
        file = file.numpy().decode("utf-8")
        lanes = get_horizontal_values_for_four_lanes(file)
        lanes = np.array(lanes)

        gt = np.zeros(self.cls_shape)

        for y in range(self.cls_shape[1]):
            yf = y / self.cls_shape[1]
            ly = int(lanes.shape[1] * yf)

            lx = lanes.T[ly, :]

            invalids = np.where(lx == -1)

            xf = lx / LLAMAS_SHAPE[1]
            x = np.round(xf * self.cls_shape[0] - 1).astype(np.int)

            x[invalids] = self.cls_shape[0] - 1

            # delete unused lanes
            x = np.delete(x, (0, 3))

            for i in range(len(x)):
                gt[x[i], y, i] = 1

        return gt

    def process_json(self, json):
        img = tf.py_function(self.read_image, [json], Tout=DTYPE)
        grid = tf.py_function(self.generate_grid_llamas, [json], Tout=DTYPE)

        img = tf.reshape(img, shape=self.image_shape)
        grid = tf.reshape(grid, shape=self.cls_shape)
        return img, grid


llds = LlamasDS(CLS_SHAPE, INPUT_SHAPE)

llamas_train_ds = tf.data.Dataset.list_files(os.path.join(BASE_PATH, 'train', '*/*.json'))
llamas_valid_ds = tf.data.Dataset.list_files(os.path.join(BASE_PATH, 'valid', '*/*.json'))

llamas_train_ds = llamas_train_ds.shuffle(SHUFFLE_SIZE)
llamas_train_ds = llamas_train_ds.take(TAKE_LLAMAS_MAX)
llamas_train_ds = llamas_train_ds.map(llds.process_json, num_parallel_calls=PARALLEL_JOBS)

llamas_valid_ds = llamas_valid_ds.shard(num_shards=2, index=0)
llamas_test_ds = llamas_valid_ds.shard(num_shards=2, index=1)

llamas_valid_ds = llamas_valid_ds.map(llds.process_json, num_parallel_calls=PARALLEL_JOBS)
llamas_test_ds = llamas_test_ds.map(llds.process_json, num_parallel_calls=PARALLEL_JOBS)
ufm = tf.keras.models.load_model('trained_model/ultrafast.tf', compile=False)
ufm.compile(optimizer='adam', loss=losses.ultrafast_loss, metrics=[metrics.ultrafast_accuracy])
for img, mask in llamas_test_ds.take(5):
    s = time.time()
    pred = ufm.predict(tf.expand_dims(img, axis=0))
    e = time.time()

    print(e - s)

    pred = pred[0]

    m = tf.zeros((CLS_SHAPE[0], CLS_SHAPE[1], 3 - CLS_SHAPE[2]), dtype=DTYPE)
    m = tf.concat([pred, m], -1)

    m2 = tf.zeros((CLS_SHAPE[0], CLS_SHAPE[1], 3 - CLS_SHAPE[2]), dtype=DTYPE)
    m2 = tf.concat([mask, m2], -1)

    display([img, np.flipud(np.rot90(m2)), np.flipud(np.rot90(m))])