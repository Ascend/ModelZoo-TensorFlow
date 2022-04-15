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
from preprocess import preprocess_for_train
import efficientnet_builder
import efficientnet_model

from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.npu_init import *
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
import time


def isin(a, s):
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
    h = tf.cast(parsed_features['height'], tf.int32)
    w = tf.cast(parsed_features['width'], tf.int32)
    c = tf.cast(parsed_features['channels'], tf.int32)
    images = tf.reshape(images, [h, w, 3])
    images = tf.cast(images, tf.float32)
    images = images / 255.0
    images = preprocess_for_train(images, 224, 224, None)
    return images, label





def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy


def training_op(log, labels, ):
    one_hot_labels = tf.one_hot(labels, 1000)
    cross_entropy = tf.losses.softmax_cross_entropy(logits=log, onehot_labels=one_hot_labels)
    loss = tf.reduce_mean(cross_entropy, name='loss')
    correct = tf.nn.in_top_k(log, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, "float"))
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    tf.summary.scalar('./loss', loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        op = optimizer.minimize(loss)
    return op, loss, accuracy
    # loss_scaling = 2 ** 12
    # grads = optimizer.compute_gradients(loss * loss_scaling)
    # # grads = [(grad / loss_scaling, var) for grad, var in grads]
    # for i, (grad, var) in enumerate(grads):
    #     if grad is not None:
    #         grads[i] = (grad / loss_scaling, var)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #     train_op = optimizer.apply_gradients(grads)
    #     return train_op, loss, accuracy


# is_training = True
# batch_size = 160
# epochs = 1
# image_num = 1281167


# TMP_DATA_PATH = './data/'
# TMP_MODEL_PATH = './model/'
# TMP_LOG_PATH = './log/'
# TMP_WEIGHTS_PATH = './weights/'

# WEIGHTS_MODEL = TMP_WEIGHTS_PATH + "model.ckpt"

#add
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--is_training",type=bool, default=True)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--image_num", type=int, default=1281167)
parser.add_argument("--batch_size", type=int, default=90)
parser.add_argument("--TMP_DATA_PATH", type=str, default='./data')
parser.add_argument("--TMP_MODEL_PATH", type=str, default='./model')
parser.add_argument("--TMP_LOG_PATH", type=str, default='./log')
parser.add_argument("--TMP_WEIGHTS_PATH", type=str, default='/data1/NRE_Check/wx1056345/ID1220_Efficientnet_V2/weights')

FLAGS = parser.parse_args()

WEIGHTS_MODEL = FLAGS.TMP_WEIGHTS_PATH + "/" + "model.ckpt"

def tf_data_list(tf_data_path):
    filepath = tf_data_path
    tf_data_list = []
    file_list = os.listdir(filepath)
    for i in file_list:
        tf_data_list.append(os.path.join(filepath, i))
    print("-----------------------------------------------------")
    print(tf_data_list)
    return tf_data_list


dataset = tf.data.TFRecordDataset(tf_data_list(FLAGS.TMP_DATA_PATH + "/" +"train_tf"))
dataset = dataset.map(_parse_read, num_parallel_calls=1)

if FLAGS.is_training:
    dataset = dataset.shuffle(FLAGS.batch_size * 6)
    dataset = dataset.repeat(FLAGS.epochs)
else:
    dataset = dataset.repeat(1)

dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
iterator = dataset.make_one_shot_iterator()
images_batch, labels_batch = iterator.get_next()
print(images_batch, labels_batch)

inputx = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 224, 224, 3], name="inputx")
inputy = tf.placeholder(tf.int64, name="inputy")
out, model_endpoint = efficientnet_builder.build_model(
    inputx,
    model_name="efficientnet-b0",
    training=FLAGS.is_training,
    override_params=None)

train_op, train_loss, train_val = training_op(out, inputy)
test_acc = evaluation(out, inputy)

config = tf.ConfigProto(allow_soft_placement=True)
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训练
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, WEIGHTS_MODEL)

saver_train = tf.train.Saver(max_to_keep=50)

print("Training start....")
try:
    tf.logging.set_verbosity(tf.logging.INFO)
    global_step = 0
    perf_lsit=[]
    fps_list=[]
    for epoch in range(FLAGS.epochs):
        for step in range(int(FLAGS.image_num / FLAGS.batch_size)):    #900/90=10
            star_time = time.time()
            x_in, y_in = sess.run([images_batch, labels_batch])
            y_in = np.squeeze(y_in, 1)
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_val],
                                                     feed_dict={inputx: x_in, inputy: y_in})
            if (step + 1) % 1 == 0:
                if step > 0:   #去掉第一次不稳定数据
                    perf = time.time() - star_time
                    perf_lsit.append(perf)
                    perf_ = np.mean(perf_lsit)
                    fps = FLAGS.batch_size / perf 
                    fps_list.append(fps)
                    fps_ = np.mean(fps_list)
                    print('Epoch %d step %d train loss = %.4f train accuracy = %.2f%%  time:  %.4f  fps: %.4f' % (
                        epoch + 1, step + 1, tra_loss, tra_acc * 100.0, perf_, fps_))
        checkpoint_path = os.path.join(FLAGS.TMP_MODEL_PATH, "m.ckpt")
        saver_train.save(sess, checkpoint_path, global_step=epoch)
except tf.errors.OutOfRangeError:
    print('epoch limit reached')
finally:
    print("Training Done")
    sess.close()
