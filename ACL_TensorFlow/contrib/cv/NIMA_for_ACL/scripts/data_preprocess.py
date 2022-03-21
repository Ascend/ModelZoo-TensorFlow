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
from tensorflow.python.platform import gfile
import tensorflow as tf
from tensorflow.python.keras import backend as K
import sys

dataset_base = sys.argv[1]
output_dir = sys.argv[2]

base_images_path = os.path.join(dataset_base,'image')
ava_dataset_path = os.path.join(dataset_base,'AVA.txt')

IMAGE_SIZE = 224

files = glob.glob(base_images_path + "*.jpg")
files = sorted(files)

train_image_paths = []
train_scores = []

image_num = 5000
batch_size = 1

print("Loading training set and val set")
print(ava_dataset_path)
with open(ava_dataset_path, mode='r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        token = line.split()
        id = int(token[1])
        print(id)
        values = np.array(token[2:12], dtype='float32')
        values /= values.sum()

        file_path = os.path.join(base_images_path , str(id) + '.jpg')
        print(file_path)
        if os.path.exists(file_path):
            train_image_paths.append(file_path)
            train_scores.append(values)



train_image_paths = np.array(train_image_paths)
train_scores = np.array(train_scores, dtype='float32')

val_image_paths = train_image_paths
val_scores = train_scores

print("--------------len-----------------",len(train_image_paths))
print('Val set size : ', val_image_paths.shape, val_scores.shape)
print('Train and validation datasets ready !')


def parse_data_without_augmentation(filename, scores):
    '''
    Loads the image file without any augmentation. Used for validation set.

    Args:
        filename: the filename from the record
        scores: the scores from the record

    Returns:
        an image referred to by the filename and its scores
    '''
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores


    '''
    Creates a python generator that loads the AVA dataset images without random data
    augmentation and generates numpy arrays to feed into the Keras model for training.

    Args:
        batchsize: batchsize for validation set

    Returns:
        a batch of samples (X_images, y_scores)
    '''
    # with tf.Session() as sess:
sess = K.get_session() 




val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_scores))
val_dataset = val_dataset.map(parse_data_without_augmentation)

val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.repeat()
# val_dataset = val_dataset.batch(batchsize, drop_remainder=True)
val_iterator = val_dataset.make_initializable_iterator()
val_batch = val_iterator.get_next()
sess.run(val_iterator.initializer)

xdata_path = os.path.join(output_dir,'xdata')
ydata_path = os.path.join(output_dir,'ydata')
if not os.path.exists(xdata_path):
    os.makedirs(xdata_path)
if not os.path.exists(ydata_path):
    os.makedirs(ydata_path)

label  = []
for step in range(int(image_num/batch_size)):
    X_batch, y_batch = sess.run(val_batch)
    X_batch.tofile(os.path.join(xdata_path,str(step)+".bin"))
    label += y_batch.tolist()
label = np.array(label)
np.save(os.path.join(ydata_path,"imageLabel.npy"), label)
