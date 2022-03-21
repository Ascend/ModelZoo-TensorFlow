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
import os
import re
import tensorflow as tf
import numpy as np
import sys
from math import *
from preprocess import preprocess_for_eval
import efficientnet_builder
import efficientnet_model

from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.npu_init import *
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow



def isin(a,s):
    for i in a:
        if i in s:
            return True
    return False

def _parse_read(example_proto):
    features = {"image": tf.FixedLenFeature([], tf.string, default_value=""),
                "height": tf.FixedLenFeature([], tf.int64, default_value=[0]),
                "width": tf.FixedLenFeature([], tf.int64, default_value=[0]),
                "channels": tf.FixedLenFeature([], tf.int64, default_value=[3]),
                "colorspace": tf.FixedLenFeature([], tf.string, default_value=""),
                "img_format": tf.FixedLenFeature([], tf.string, default_value=""),
                "label": tf.FixedLenFeature([1], tf.int64, default_value=[0]),
                "bbox_xmin": tf.VarLenFeature(tf.float32),
                "bbox_xmax": tf.VarLenFeature(tf.float32),
                "bbox_ymin": tf.VarLenFeature(tf.float32),
                "bbox_ymax": tf.VarLenFeature(tf.float32),
                "text": tf.FixedLenFeature([], tf.string, default_value=""),
                "filename": tf.FixedLenFeature([], tf.string, default_value="")
                }

    parsed_features = tf.parse_single_example(example_proto, features)
    label = parsed_features["label"]
    images = tf.image.decode_jpeg(parsed_features["image"])
    h = tf.cast(parsed_features['height'], tf.int64)
    w = tf.cast(parsed_features['width'], tf.int64)
    c = tf.cast(parsed_features['channels'], tf.int64)
    images = tf.reshape(images, [h, w, 3])
    images = tf.cast(images, tf.float32)
    images = images/255.0
    images = preprocess_for_eval(images, 224, 224, 0.83)
    return images, label

def _parse_augmentation(image,label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image,label


def evaluation(logits, labels):
    # acc calc
    # args：logits，net's output; labels，the real value，0 or 1
    # return：accuracy, The average accuracy of the current step, that is, how many
    # images in these batches are correctly classified.
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels,1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy',accuracy)
    return  accuracy


def training_op( log,label):
    # loss
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=log, labels=label, name='entropy_per_example')
    loss = tf.reduce_mean(cross_entropy, name='loss')
    # accuracy
    correct = tf.nn.in_top_k(log, label, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, "float"))
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        op = optimizer.minimize(loss)
        return op,loss,accuracy



batch_size = 50
image_num = 50000
tf_data = []
TMP_DATA_PATH = './data/valid'
TMP_MODEL_PATH = './model/m.ckpt-0'
TMP_LOG_PATH = './log/'

data_list = os.listdir(TMP_DATA_PATH)
for i in  data_list:
    tf_data.append(os.path.join(TMP_DATA_PATH,i))
print(tf_data)
dataset = tf.data.TFRecordDataset(tf_data)
dataset = dataset.map(_parse_read,num_parallel_calls=2)
dataset = dataset.repeat(1)
dataset = dataset.batch(batch_size, drop_remainder=True)
iterator = dataset.make_one_shot_iterator()
images_batch, labels_batch = iterator.get_next()

inputx = tf.placeholder(tf.float32, shape=[batch_size, 224, 224, 3], name="inputx")
inputy = tf.placeholder(tf.int64, name="inputy")
out,model_endpoint= efficientnet_builder.build_model(
                                            inputx,
                                            model_name="efficientnet-b0",
                                            training=False,
                                            override_params=None)
train_op, train_loss, train_val = training_op(out, inputy)
test_acc = evaluation(out, inputy)

config = tf.ConfigProto(allow_soft_placement=True)
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训练
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, TMP_MODEL_PATH)

print("test start....")
try:
    total = []
    for step in range(int(image_num / batch_size)):
        x_in, y_in = sess.run([images_batch, labels_batch])
        y_in = np.squeeze(y_in, 1)
        tra_acc = sess.run([test_acc], feed_dict={inputx: x_in, inputy: y_in})
        total.append(tra_acc)
    print("Top-1 acc: %.3f"%np.mean(total))
except tf.errors.OutOfRangeError:
    print('epoch limit reached')
finally:
    print("test Done")
    sess.close()