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

# -*- coding: UTF-8 -*-
import tensorflow as tf
import os
import numpy as np
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

is_training = False
batch_size = 
epochs = 1
image_num = 1449

def read_data(tf_file, is_training):
    def _parse_read(tfrecord_file):
        features = {
            'image':
                tf.io.FixedLenFeature((), tf.string),
            "label":
                tf.io.FixedLenFeature((), tf.string),
            "mask":
                tf.io.FixedLenFeature((), tf.string),
            'height':
                tf.io.FixedLenFeature((), tf.int64),
            'width':
                tf.io.FixedLenFeature((), tf.int64),
            'channels':
                tf.io.FixedLenFeature((), tf.int64)
        }
        parsed = tf.io.parse_single_example(tfrecord_file, features)
        image = tf.decode_raw(parsed['image'], tf.uint8)
        image = tf.reshape(image, [parsed['height'], parsed['width'], parsed['channels']])
        label = tf.decode_raw(parsed['label'], tf.uint8)
        label = tf.reshape(label, [parsed['height'], parsed['width'], parsed['channels']])
        mask = tf.decode_raw(parsed['mask'], tf.uint8)
        mask = tf.reshape(mask, [parsed['height'], parsed['width'], 24])
        label = label[:, :, 0:1]
        mask = mask[:, :, 0:1]

        image, label, mask = _augmentation(image, label, mask, parsed['height'], parsed['width'])
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int64)
        mask = tf.cast(mask, tf.float32)
        image, label, mask = _preprocess(image, label, mask)
        return image, label[:, :, 0], mask[:, :, 0]

    def _augmentation(image, label, mask, h, w):
        image = tf.image.resize_images(image, size=[512, 512], method=0)
        image = tf.cast(image, tf.uint8)
        label = tf.image.resize_images(label, size=[64, 64], method=1)
        mask = tf.image.resize_images(mask, size=[64, 64], method=1)
        # 随机翻转
        with tf.Session() as sess:
            rand_value = tf.random.uniform(()).eval()
     
        if rand_value > 0.5:
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
            mask = tf.image.flip_left_right(mask)
        return image, label, mask

    def _preprocess(image, label, mask):
        image = image - [122.67891434, 116.66876762, 104.00698793]
        image = image / 255.
        return image, label, mask

    dataset = tf.data.TFRecordDataset(tf_file, num_parallel_reads=4)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=batch_size * 10))
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(map_func=_parse_read, batch_size=batch_size, drop_remainder=True,
                                      num_parallel_calls=2))
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch, masks_batch = iterator.get_next()
    return images_batch, labels_batch

def Spectral_distance():
    spectral_distance = np.zeros([10, 10])
    for i in range(10):
        for j in range(10):
            spectral_distance[i, j] = (i - 5) ** 2 + (j - 5) ** 2
    spectral_distance = (spectral_distance / 50).astype(np.float32)
    spectral_distance = np.expand_dims(spectral_distance, axis=-1)
    return spectral_distance

if __name__ == '__main__':

    tf_data_path = sys.argv[1]
    output_path = sys.argv[2]

    data_path = os.path.join(output_path,'data')
    distance_path = os.path.join(output_path,'distance')
    label_path = os.path.join(output_path,'label')

    clear = True
    if clear:
        os.system("rm -rf {}".format(data_path))
        os.system("rm -rf {}".format(distance_path))
        os.system("rm -rf {}".format(label_path))
    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    if not os.path.isdir(distance_path):
        os.makedirs(distance_path)
    if not os.path.isdir(label_path):
        os.makedirs(label_path)

    distance = Spectral_distance()
    images_batch, labels_batch = read_data(tf_data_path,is_training)
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    image = []
    label = []
    for step in tqdm(range(int(image_num / batch_size))):
        x_in, y_in = sess.run([images_batch, labels_batch])
        label.append(y_in)
        x_in.tofile(os.path.join(data_path, str(step).zfill(6)+".bin")
        distance.tofile(os.path.join(distance_path, str(step).zfill(6)+".bin"))
    label = np.array(label)
    np.save(os.path.join(label_path, "label.npy"", label)
    print("[Info]  data preprocess done!")
