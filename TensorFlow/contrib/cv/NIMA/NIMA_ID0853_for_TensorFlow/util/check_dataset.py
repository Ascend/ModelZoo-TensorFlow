
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
import os
import glob

import tensorflow as tf

'''
Checks all images from the AVA dataset if they have corrupted jpegs, and lists them for removal.

Removal must be done manually !
'''

base_images_path = r'../AVA_dataset/image/images/'
ava_dataset_path = r'../AVA_dataset/AVA.txt'


IMAGE_SIZE = 128
BASE_LEN = len(base_images_path) - 1

files = glob.glob(base_images_path + "*.jpg")
files = sorted(files)

train_image_paths = []
train_scores = []

print("Loading training set and val set")
with open(ava_dataset_path, mode='r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        token = line.split()
        id = int(token[1])

        values = np.array(token[2:12], dtype='float32')
        values /= values.sum()

        file_path = base_images_path + str(id) + '.jpg'
        if os.path.exists(file_path):
            train_image_paths.append(file_path)
            train_scores.append(values)

        count = 255000 // 20
        if i % count == 0 and i != 0:
            print('Loaded %0.2f of the dataset' % (i / 255000. * 100))


train_image_paths = np.array(train_image_paths)
train_scores = np.array(train_scores, dtype='float32')

val_image_paths = train_image_paths[-5000:]
val_scores = train_scores[-5000:]
train_image_paths = train_image_paths[:-5000]
train_scores = train_scores[:-5000]

def parse_data(filename):
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

sess = tf.Session()
with sess.as_default():
    sess.run(tf.global_variables_initializer())

    count = 0
    fn = tf.placeholder(dtype=tf.string)
    img = parse_data(fn)

    for path in train_image_paths:
        try:
            sess.run(img, feed_dict={fn: path})
        except Exception as e:
            print(path, "failed to load !")
            print()
            count += 1

    print(count, "images failed to load !")

print("All done !")

"""
Had to delete file : 440774.jpg and remove row from AVA.txt
Had to delete file : 179118.jpg and remove row from AVA.txt
Had to delete file : 371434.jpg and remove row from AVA.txt
Had to delete file : 277832.jpg and remove row from AVA.txt
Had to delete file : 230701.jpg and remove row from AVA.txt
Had to delete file : 729377.jpg and remove row from AVA.txt
"""