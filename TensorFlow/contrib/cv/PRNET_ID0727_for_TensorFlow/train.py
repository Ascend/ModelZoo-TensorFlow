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
import argparse
import tensorflow as tf
import cv2
import random
from predictor import resfcn256
import math
import glob
from datetime import datetime
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import time

class TrainData(object):

    def __init__(self, rootdir):
        super(TrainData, self).__init__()
        self.rootdir = rootdir
        self.train_data_list = []
        self.readTrainData()
        self.index = 0
        self.num_data = len(self.train_data_list)

    def readTrainData(self):
        sub_dir_list = os.listdir(self.rootdir)
        for item in sub_dir_list:
            img_list = glob.glob(os.path.join(self.rootdir, item, "*.jpg"))
            for img_path in img_list:
                npy_path = img_path.replace('jpg', 'npy')
                self.train_data_list.append([img_path, npy_path])
            random.shuffle(self.train_data_list)

    def getBatch(self, batch_list):
        batch = []
        imgs = []
        labels = []
        for item in batch_list:
            img = cv2.imread(item[0])
            label = np.load(item[1])

            img_array = np.array(img, dtype=np.float32)
            imgs.append(img_array / 256.0 / 1.1)

            label_array = np.array(label, dtype=np.float32)
            labels.append(label_array / 256 / 1.1)

        batch.append(imgs)
        batch.append(labels)

        return batch

    def __call__(self, batch_num):
        if (self.index + batch_num) <= self.num_data:
            batch_list = self.train_data_list[self.index:(self.index + batch_num)]
            batch_data = self.getBatch(batch_list)
            self.index += batch_num

            return batch_data
        else:
            self.index = 0
            random.shuffle(self.train_data_list)
            batch_list = self.train_data_list[self.index:(self.index + batch_num)]
            batch_data = self.getBatch(batch_list)
            self.index += batch_num

            return batch_data


def main(args):

    batch_size = args.batch_size
    epochs = args.epochs
    end_data_num = args.end_data_num
    root_train_data_dir = args.root_train_data_dir
    model_path = args.model_path

    save_dir = args.checkpoint
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Training data
    data = TrainData(root_train_data_dir)

    begin_epoch = 0
    if os.path.exists(model_path + '.data-00000-of-00001'):
        begin_epoch = int(model_path.split('_')[-1]) + 1

    epoch_iters = data.num_data / batch_size
    global_step = tf.Variable(epoch_iters * begin_epoch, trainable=False)

    decay_steps = 5 * epoch_iters

    learning_rate = tf.train.exponential_decay(args.learning_rate, global_step,
                                               decay_steps, 0.5, staircase=True)

    x = tf.placeholder(tf.float32, shape=[batch_size, 256, 256, 3])
    label = tf.placeholder(tf.float32, shape=[batch_size, 256, 256, 3])

    # Train net
    net = resfcn256(256, 256)
    x_op = net(x, is_training=True)

    # Loss
    weights = cv2.imread((os.path.dirname(__file__) + "Data/uv-data/weight_mask_final.jpg"))  # [256, 256, 3]
    weights_data = np.zeros([1, 256, 256, 3], dtype=np.float32)
    weights_data[0, :, :, :] = weights  # / 16.0
    loss = tf.losses.mean_squared_error(label, x_op, weights_data)

    # This is for batch norm layer
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                            beta1=0.9, beta2=0.999, epsilon=1e-08,
                                            use_locking=False).minimize(loss, global_step=global_step)

    # sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  
    sess = tf.Session(config=config)
    sess.run(init)

    # if os.path.exists(model_path + '.data-00000-of-00001'):
    #     tf.train.Saver(net.vars).restore(sess, model_path)

    saver = tf.train.Saver(var_list=tf.global_variables())
    save_path = model_path

    # Begining train
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fp_log = open("log_" + time_now + ".txt", "w")
    iters_total_each_epoch = int(math.ceil(1.0 * data.num_data / batch_size))-end_data_num
    for epoch in range(begin_epoch, epochs):
        for iters in range(iters_total_each_epoch):
            start = time.time()
            
            batch = data(batch_size)
            loss_res, _, global_step_res, learning_rate_res = sess.run(
                [loss, train_step, global_step, learning_rate], feed_dict={x: batch[0], label: batch[1]})

            end = time.time()
            runtime = end-start
            log_line = 'global_step:%d : iters:%d  /  epoch:%d,learning rate:%f   loss:%f   iter_time:%f' % (global_step_res, iters, epoch, learning_rate_res, loss_res, runtime)

            print(log_line)
            fp_log.writelines(log_line + "\n")

        saver.save(sess=sess, save_path=save_path + '_' + str(epoch))
    tf.train.Saver().save(sess, "checkpoint/model.ckpt")
    tf.train.write_graph(sess.graph, './checkpoint', 'graph.pbtxt', as_text=True)

    sess.close()
    fp_log.close()


if __name__ == '__main__':

    code_dir = os.path.dirname(__file__)
    par = argparse.ArgumentParser(
        description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    par.add_argument('--root_train_data_dir', default='Dataset/TrainData/', type=str, help='The training data file')
    par.add_argument('--learning_rate', default=0.0002, type=float, help='The learning rate')
    par.add_argument('--epochs', default=10, type=int, help='Total epochs')
    par.add_argument('--end_data_num', default=0, type=int, help='contrl step')
    par.add_argument('--batch_size', default=32, type=int, help='Batch sizes')
    par.add_argument('--checkpoint', default='checkpoint/', type=str, help='The path of checkpoint')
    par.add_argument('--model_path', default='checkpoint/256_256_resfcn256_weight', type=str, help='modelpath')
    main(par.parse_args())
