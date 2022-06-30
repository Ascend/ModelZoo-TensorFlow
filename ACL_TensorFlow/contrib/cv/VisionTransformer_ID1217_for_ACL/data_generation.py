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
import tensorflow as tf
import os
import time
import numpy as np
import pickle
from tqdm import tqdm

image_size = 384

DATA_CACHE_PATH = "./dataset"

HPARAMS = {
    "batch_size": 4,
    "image_size": 384,
    'learning_rate': 0.001,
}

def read_data(filename, training):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    if training:
        images = dict[b'data'].reshape([50000, 3, 32, 32])
    else:
        images = dict[b'data'].reshape([10000, 3, 32, 32])
    images = np.transpose(images, [0, 2, 3, 1])
    labels = np.array(dict[b'fine_labels'])

    def _augment(image, label):
        if np.random.rand() < 0.3:
            image = tf.image.flip_left_right(image)
        if np.random.rand() < 0.3:
            image = tf.image.flip_up_down(image)
        if np.random.rand() < 0.3:
            image = tf.image.random_contrast(image, lower=0.5, upper=2)
        return image, label

    def _preprocess(image, label):
        image = tf.image.resize(image, (image_size, image_size))
        image = (image - 127.5) / 127.5
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        # ds = ds.map(_augment)
        ds = ds.map(_preprocess)
        ds = ds.shuffle(HPARAMS['batch_size'] * 10)
        ds = ds.repeat()
    else:
        ds = ds.map(_preprocess)
        ds = ds.repeat()

    ds = ds.batch(batch_size=HPARAMS['batch_size'], drop_remainder=True)
    iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
    image_batch, label_batch = iterator.get_next()
    print("work on it =================================")
    return image_batch, label_batch



images_batch, labels_batch = read_data(filename=DATA_CACHE_PATH+"/train", training=True)
val_image_batch, val_labels_batch = read_data(filename=DATA_CACHE_PATH+"/test", training=False)

inputx = tf.compat.v1.placeholder(
    tf.float32, shape=[HPARAMS['batch_size'], HPARAMS['image_size'], HPARAMS['image_size'], 3], name="inputx")

inputy = tf.compat.v1.placeholder(
    tf.int64, shape=[HPARAMS['batch_size'], ], name="inputy")

inputTrain = tf.compat.v1.placeholder(
    tf.bool, name='training')


def eval(pred, label):
    prediction = np.argmax(pred, 1).tolist()
    return calc(prediction, label)


def calc(prediction, label):
    a = [prediction[i] == label[i] for i in range(len(prediction))]
    return sum(a) / len(a)

config = tf.ConfigProto(allow_soft_placement=True)
config = tf.ConfigProto()
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

s = time.time()

for epoch in range(1,2):
    # train
    # break
    label_col = []
    pred_col = []
    # labels = []
    labels = []
    
    for step in tqdm(range(10000 // HPARAMS['batch_size'])):
        x_in, y_in = sess.run([val_image_batch, val_labels_batch])
        x_in.tofile("./bindataset/{}.bin".format(step))
        labels += y_in.tolist()
    
    np.save("./label.npy",labels)
