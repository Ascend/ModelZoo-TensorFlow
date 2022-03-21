# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import tensorflow as tf
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import random
from resnest import deeplabv3_resnest50
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def read_data(tf_file, batch_size, is_training):
    def _parse_read(tfrecord_file):
        features = {
            'image':
                tf.io.FixedLenFeature((), tf.string),
            "label":
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
        image = tf.reshape(
            image, [parsed['height'], parsed['width'], parsed['channels']])
        label = tf.decode_raw(parsed['label'], tf.uint8)
        label = tf.reshape(label, [parsed['height'], parsed['width'], 1])
        combined = tf.concat([image, label], axis=-1)
        combined = tf.random_crop(combined, (img_H, img_W, 4))
        image = combined[:, :, 0:3]
        label = combined[:, :, 3:4]
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int64)
        return image, label[:, :, 0]

    def _preprocess(image, label):
        image = image / 255.
        image = image - [0.406, 0.456, 0.485]
        image = image/[0.225, 0.224, 0.229]
        return image, label

    dataset = tf.data.TFRecordDataset(tf_file)
    dataset = dataset.map(_parse_read, num_parallel_calls=2)
    if is_training:
        dataset = dataset.map(_preprocess, num_parallel_calls=2)
        dataset = dataset.shuffle(batch_size * 10)
        dataset = dataset.repeat()
    else:
        dataset = dataset.map(_preprocess, num_parallel_calls=2)
        dataset = dataset.repeat(1)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    iterator = dataset.make_one_shot_iterator()
    images_batch, labels_batch = iterator.get_next()
    print(images_batch, labels_batch)
    return images_batch, labels_batch


def get_matrix(predict, label, num_classes):
    mask = (label >= 0) & (label <= num_classes)
    label = num_classes * label[mask].astype('int') + predict[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    return confusion_matrix


def get_result(confusion_matrix):
    
    Pixel_acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    
    MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                                        np.diag(confusion_matrix))
    print(MIoU)
    MIoU = np.nanmean(MIoU)
    
    Mean_acc = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    Mean_acc = np.nanmean(Mean_acc)
    
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
        np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
        np.diag(confusion_matrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return Pixel_acc, Mean_acc, MIoU, FWIoU


is_training = False
gpu = 1

# ap = argparse.ArgumentParser()
# ap.add_argument("-h", "--img_H", required=False,
                # default=1024, help="image height")
# ap.add_argument("-w", "--img_W", default=2048,
                # required=False, help="image width")
# ap.add_argument("-m", '--model_path',
                # default="./npu_resnest/resnest.ckpt", help="model path")
# ap.add_argument("-b", '--batch_size', required=False,
                # default=1, help="batch of data")
# ap.add_argument("-c", "--num_classes", required=False,
                # default=True, help="number classes")
# ap.add_argument("-d", '--data_path', required=False,
                # default="./test_data/cityscapes_val.tfrecords", help="data path")
# args = vars(ap.parse_args())

# img_H = args.img_H
# img_W = args.img_W
# num_classes = args.num_classes
# batch_size = args.batch_size
# test_tf = args.data_path
# model = args.model_path
img_H = 1024
img_W = 2048
num_classes = 19
batch_size = 1
test_tf = "./test_data/cityscapes_val.tfrecords"
model = "./npu_resnest/resnest.ckpt-182"


images_batch, labels_batch = read_data(test_tf, batch_size, is_training)
print("-------------------------------", images_batch)
print("-------------------------------", labels_batch)

os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % gpu
with tf.device("/gpu:%d" % gpu):
    inputx = tf.placeholder(
        tf.float32, shape=[batch_size, img_H, img_W, 3], name="inputx")
    inputy = tf.placeholder(
        tf.int64, shape=[batch_size, img_H, img_W],  name="inputy")
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        out = deeplabv3_resnest50(inputx, is_training, [img_H, img_W])
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, model)
        c_matrix = np.zeros((num_classes, num_classes))
        print("Test start....")
        try:
            for step in range(100):
                print(step)
                x_in, y_in = sess.run([images_batch, labels_batch])
                prediction = sess.run(out, feed_dict={inputx: x_in})
                pre = np.argmax(prediction, axis=-1)
                c_matrix += get_matrix(pre, y_in, num_classes=num_classes)
        except tf.errors.OutOfRangeError:
            print('epoch limit reached')
        finally:
            p_acc, m_acc, miou, fmiou = get_result(c_matrix)
            print(p_acc, m_acc, miou, fmiou)
            print("miou", miou)
            print(args['model'], "--------------------------Test Done")
