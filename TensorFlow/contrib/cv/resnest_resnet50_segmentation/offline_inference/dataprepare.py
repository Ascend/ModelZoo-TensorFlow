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
from tqdm import tqdm

# import moxing as mox
# from npu_bridge.estimator import npu_ops
# from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
# from tensorflow.python.framework import graph_util
# from tensorflow.python import pywrap_tensorflow
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
    # sourcery skip: inline-immediately-returned-variable
    mask = (label >= 0) & (label <= num_classes)
    label = num_classes * label[mask].astype('int') + predict[mask]
    count = np.bincount(label, minlength=num_classes ** 2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    return confusion_matrix


def get_result(confusion_matrix):
    # pixel accuracy
    Pixel_acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    # mean iou
    MIoU = np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                                        np.diag(confusion_matrix))
    print(MIoU)
    MIoU = np.nanmean(MIoU)
    # mean pixel accuracy
    Mean_acc = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    Mean_acc = np.nanmean(Mean_acc)
    # frequncey iou
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
        np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
        np.diag(confusion_matrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return Pixel_acc, Mean_acc, MIoU, FWIoU

is_training = False
num_classes = 19
img_H = 1024
img_W = 2048


if __name__ == '__main__':
    batch_size = 1
    data_path = sys.argv[0]
    output_path = sys.argv[1]
    output_path += "/"
    image_num = 500

    clear = True
    if clear:
        os.system("rm -rf "+output_path+"data")
        os.system("rm -rf "+output_path+"label")
    if not os.path.isdir(output_path + "data"):
        os.makedirs(output_path+"data")
    if not os.path.isdir(output_path + "label"):
        os.makedirs(output_path+"label")
    images_batch, labels_batch = read_data(data_path, batch_size, is_training)
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    image = []
    label = []
    for step in tqdm(range(int(image_num / batch_size))):
        x_in, y_in = sess.run([images_batch, labels_batch])
        # label.append(y_in)
        np.save(output_path + "label/{}.npy".format(step), y_in)
        x_in.tofile(output_path+"data/"+str(step)+".bin")
    # label = np.array(label)
    # np.save(output_path + "label/label.npy", label)
    print("[info]  data bin ok")
