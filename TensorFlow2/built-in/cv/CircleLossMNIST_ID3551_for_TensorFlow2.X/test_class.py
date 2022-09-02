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
#
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras import backend as k
import matplotlib.pyplot as plt
import numpy as np
import argparse
import npu_device
import ast
import time


class ConvBnRelu(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides):
        super(ConvBnRelu, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.99,)

    def call(self, inputs, training=None, mask=None):
        inputs = self.conv(inputs)
        inputs = self.bn(inputs, training=training)
        inputs = tf.nn.relu(inputs)
        return inputs


class CicleLoss(tf.keras.Model):
    def __init__(self, category=10, margin=0., reweight=5.):
        super(CicleLoss, self).__init__()
        self.reweight = reweight
        self.pos_margin = 1. - margin
        self.neg_margin = margin
        self.category = category
        self.op = 1. + self.pos_margin
        self.on = - self.neg_margin
        """需要使用归一化产生最优正分和最优负分吗？"""
        self.dense = tf.keras.layers.Dense(units=category, activation=tf.nn.softmax)

    def call(self, inputs, training=None, mask=None):
        if training:
            inputs, label = inputs
            score = self.dense(inputs)
            label = tf.one_hot(label, depth=self.category, dtype=tf.float32)
            neg = tf.exp(self.reweight*tf.nn.relu(score-self.on)*(score-self.neg_margin))
            pos = tf.exp(-self.reweight*tf.nn.relu(self.op-score)*(score-self.pos_margin))
            pos = k.sum(pos*label, axis=-1)
            neg = k.sum(neg*(1.-label), axis=-1)
            loss = tf.math.log1p(pos * neg)
            return k.mean(loss)
        else:
            score = self.dense(inputs)
            return tf.argmax(score, axis=-1)


class classfier(tf.keras.Model):
    def __init__(self):
        super(classfier, self).__init__()
        self.conv0 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same")
        self.conv1 = ConvBnRelu(filters=64, kernel_size=3, strides=1)
        self.conv2 = ConvBnRelu(filters=128, kernel_size=3, strides=1)
        self.conv3 = ConvBnRelu(filters=256, kernel_size=3, strides=2)
        self.conv4 = ConvBnRelu(filters=256, kernel_size=3, strides=1)
        self.conv5 = ConvBnRelu(filters=256, kernel_size=3, strides=1)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training=None, mask=None):
        inputs = tf.nn.relu(self.conv0(inputs))
        inputs = self.conv1(inputs, training=training)
        inputs = self.conv2(inputs, training=training)
        inputs = self.conv3(inputs, training=training)
        inputs = self.conv4(inputs, training=training)
        inputs = self.conv5(inputs, training=training)
        inputs = self.pool(inputs)
        # inputs = self.latten_compress(inputs)
        return inputs

def get_args():
    parser = argparse.ArgumentParser("please input args")
    parser.add_argument("--train_epochs", type=int, default=50, help="epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("--data_path", type=str, default="./data", help="data_path")
    parser.add_argument('--log_steps', type=int, default=200, help='log steps')
    parser.add_argument('--model_dir', type=str, default='./model/', help='save model dir')
    parser.add_argument('--static', action='store_true', default=False, help='static input shape, default is False')
    parser.add_argument('--learning_rate', type=int, default=5e-4, help='Learning rate for training')
    #===============================NPU Migration=========================================
    parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='precision mode')
    parser.add_argument('--over_dump', dest='over_dump', type=ast.literal_eval, help='if or not over detection, default is False')
    parser.add_argument('--data_dump_flag', dest='data_dump_flag', type=ast.literal_eval, help='data dump flag, default is False')
    parser.add_argument('--data_dump_step', default="10", help='data dump step, default is 10')
    parser.add_argument('--profiling', dest='profiling', type=ast.literal_eval,help='if or not profiling for performance debug, default is False')
    parser.add_argument('--profiling_dump_path', default="/home/data", type=str,help='the path to save profiling data')
    parser.add_argument('--over_dump_path', default="/home/data", type=str,help='the path to save over dump data')
    parser.add_argument('--data_dump_path', default="/home/data", type=str,help='the path to save dump data')
    parser.add_argument('--use_mixlist', dest='use_mixlist', type=ast.literal_eval, help='use_mixlist flag, default is False')
    parser.add_argument('--fusion_off_flag', dest='fusion_off_flag', type=ast.literal_eval, help='fusion_off flag, default is False')
    parser.add_argument('--mixlist_file', default="ops_info.json", type=str,help='mixlist file name, default is ops_info.json')
    parser.add_argument('--fusion_off_file', default="fusion_switch.cfg", type=str,help='fusion_off file name, default is fusion_switch.cfg')
    parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval, help='autotune, default is False')
    ############多p参数##############
    parser.add_argument("--rank_size", default=1, type=int, help="rank size")
    parser.add_argument("--device_id", default=0, type=int, help="Ascend device id")

    args = parser.parse_args()
    return args


def npu_config(FLAGS):
    if FLAGS.data_dump_flag:
        npu_device.global_options().dump_config.enable_dump = True
        npu_device.global_options().dump_config.dump_path = FLAGS.data_dump_path
        npu_device.global_options().dump_config.dump_step = FLAGS.data_dump_step
        npu_device.global_options().dump_config.dump_mode = "all"

    if FLAGS.over_dump:
        npu_device.global_options().dump_config.enable_dump_debug = True
        npu_device.global_options().dump_config.dump_path = FLAGS.over_dump_path
        npu_device.global_options().dump_config.dump_debug_mode = "all"

    if FLAGS.profiling:
        npu_device.global_options().profiling_config.enable_profiling = True
        profiling_options = '{"output":"' + FLAGS.profiling_dump_path + '", \
                            "training_trace":"on", \
                            "task_trace":"on", \
                            "aicpu":"on", \
                            "aic_metrics":"PipeUtilization",\
                            "fp_point":"", \
                            "bp_point":""}'
        npu_device.global_options().profiling_config.profiling_options = profiling_options
    npu_device.global_options().precision_mode = FLAGS.precision_mode
    if FLAGS.use_mixlist and FLAGS.precision_mode=='allow_mix_precision':
        npu_device.global_options().modify_mixlist=FLAGS.mixlist_file
    if FLAGS.fusion_off_flag:
        npu_device.global_options().fusion_switch_file=FLAGS.fusion_off_file
    if FLAGS.auto_tune:
        npu_device.global_options().auto_tune_mode="RL,GA"
    npu_device.open().as_default()

@tf.function
def train_step(images, labels, model, optimizer):
    images = 2 * (tf.cast(images[..., tf.newaxis], tf.float32) - .5)
    labels = tf.cast(labels, tf.int64)
    with tf.GradientTape() as tape:
        embedding = model(images, training=True)
        loss = prediction_matrix([embedding, labels], training=True)
        # loss = k.mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,
        #                                                              labels=labels[:, 0]),
        #               axis=-1)
        grd = tape.gradient(loss, model.trainable_variables + prediction_matrix.trainable_variables)
        optimizer.apply_gradients(zip(grd, model.trainable_variables + prediction_matrix.trainable_variables))
    return loss

@tf.function
def test_step(images, labels, model):
    images = 2 * (tf.cast(images[..., tf.newaxis], tf.float32) - .5)
    labels = tf.cast(labels, tf.int64)
    embedding = model(images, training=False)
    prediction = prediction_matrix(embedding, training=False)
    return labels, prediction


if __name__ == '__main__':
    args = get_args()
    npu_config(args)
    DATA_PATH = args.data_path
    BATH_SIZE = args.batch_size
    TRAIN_EPOCHS = args.train_epochs
    MODEL_DIR = args.model_dir
    LOG_STEPS = args.log_steps
    LEARNING_RATE =args.learning_rate
    STATIC = args.static

    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(os.path.join(DATA_PATH, 'mnist.npz'))
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    traindataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    testdataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    if args.rank_size != 1:
        dataset, BATH_SIZE = npu_device.distribute.shard_and_rebatch_dataset(traindataset, args.batch_size)
    prediction_matrix = CicleLoss()
    if args.rank_size != 1:
        optimizer = npu_device.distribute.npu_distributed_keras_optimizer_wrapper(tf.keras.optimizers.Adam(LEARNING_RATE))
    else:
        optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    # Training
    model = classfier()
    if args.rank_size != 1:
        training_vars = model.trainable_variables
        npu_device.distribute.broadcast(training_vars, root_rank=0)
    if STATIC:
        traindataset = traindataset.cache().shuffle(10000, reshuffle_each_iteration=True).batch(BATH_SIZE, drop_remainder=True)
        testdataset = testdataset.batch(BATH_SIZE, drop_remainder=True)
    else:
        traindataset = traindataset.cache().shuffle(10000, reshuffle_each_iteration=True).batch(BATH_SIZE)
        testdataset = testdataset.batch(BATH_SIZE)
    start_time = time.time()
    iteration = 0
    for epoch in range(TRAIN_EPOCHS):
        step_epoch = 0
        for syncdata in traindataset:
            iteration += 1
            step_epoch += 1
            images, labels = syncdata
            loss = train_step(images, labels, model, optimizer)
        now = time.time()
        elapsed_time = now - start_time
        print('Epoch [%d/%d] (%d iteration) loss: %.3f FPS: %.3f' %(epoch+1, TRAIN_EPOCHS, iteration, loss, args.batch_size/elapsed_time*step_epoch))

        mean_acc = []
        for syncdata in testdataset:
            images, labels = syncdata
            labels, prediction = test_step(images, labels, model)
            mean_acc.append(np.mean(labels.numpy() == prediction.numpy()))
        print('val_acc', np.mean(mean_acc))
        start_time = time.time()
    model.save_weights(filepath=os.path.join(MODEL_DIR, 'tf_model'), save_format='tf')
