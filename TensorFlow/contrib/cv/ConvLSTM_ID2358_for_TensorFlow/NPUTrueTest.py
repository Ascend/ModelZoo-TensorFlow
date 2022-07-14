#
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
#
# Imports and Global Variables
# coding=UTF-8
from npu_bridge.npu_init import *
import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import cv2
import math
import warnings
import argparse
import logging
import datetime
from cell import ConvLSTMCell  # added
#precision_tool/tf_config.py
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import precision_tool.tf_config as npu_tf_config
import precision_tool.config as CONFIG

USE_CUDA = True

LSTM_HIDDEN_SIZE = 550
# TIME_STEPS = 1  changed
TIME_STEPS = 1
K = 100
#打印到GPU.log文件
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename='GPU.log',
                    filemode='a')
logger = logging.getLogger(__name__)
# Build Model
class DeepVONet(object):
    def __init__(self, args, data):
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [LSTM_HIDDEN_SIZE, LSTM_HIDDEN_SIZE]]
        # multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)  # changed
        multi_convlstm_cell = ConvLSTMCell(shape=[6, 20], filters=12, kernel=[3, 3])  # added
        rnn_inputs = []
        reuse = None
        for stacked_img in data:
            rnn_inputs.append(self.forward(stacked_img, reuse=reuse))
            reuse = True

        sess = tf.Session()
        # print(sess.run(tf.shape(stacked_img)))
        rnn_inputs = tf.transpose(rnn_inputs,perm = [1,0,2,3,4])
        print(sess.run(tf.shape(rnn_inputs)))

        # rnn_inputs = [tf.reshape(rnn_inputs[i], [-1, 20 * 6 * 1024]) for i in range(len(rnn_inputs))] #changed
        # assert rnn_inputs[0].shape == (args.bsize, 20 * 6 * 1024) #changed
        # rnn_inputs = (32,5,rnn_inputs)

        # self.outputs, self.state = tf.nn.static_rnn(cell=multi_rnn_cell, inputs=rnn_inputs, dtype=tf.float32)  # changed
        self.outputs, self.state = tf.nn.dynamic_rnn(cell=multi_convlstm_cell, inputs=rnn_inputs, dtype=tf.float32)  # added

        # print("1")
        # print(sess.run(tf.shape(self.outputs)))
        # print("2")

        self.outputs = [tf.reshape(self.outputs[i], [-1, 20 * 6 * 12]) for i in range(32)]  # added
        self.outputs = tf.reshape(self.outputs, [32, 20 * 6 * 12])  # added  (32,1440)

        # self.outputs = tf.transpose(self.outputs, perm=[1,0,2])#added
        # print("3")
        # print(sess.run(tf.shape(self.outputs)))
        # print("4")
        # assert self.outputs[0].shape == (args.bsize, LSTM_HIDDEN_SIZE)

    def forward(self, x, reuse=None):
        with tf.variable_scope("cnns", reuse=reuse):
            x = tf.layers.conv2d(
                inputs=x,
                filters=64,
                kernel_size=[7, 7],
                padding="same",
                strides=2,
                reuse=reuse,
                activation=tf.nn.relu, name='conv1')
            x = tf.layers.conv2d(
                inputs=x,
                filters=128,
                kernel_size=[5, 5],
                padding="same",
                strides=2,
                reuse=reuse,
                activation=tf.nn.relu, name='conv2')
            x = tf.layers.conv2d(
                inputs=x,
                filters=256,
                kernel_size=[5, 5],
                padding="same",
                strides=2,
                reuse=reuse,
                activation=tf.nn.relu, name='conv3')
            x = tf.layers.conv2d(
                inputs=x,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                strides=1,
                reuse=reuse,
                activation=tf.nn.relu, name='conv3_1')
            x = tf.layers.conv2d(
                inputs=x,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                strides=2,
                reuse=reuse,
                activation=tf.nn.relu, name='conv4')
            x = tf.layers.conv2d(
                inputs=x,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                strides=1,
                reuse=reuse,
                activation=tf.nn.relu, name='conv4_1')
            x = tf.layers.conv2d(
                inputs=x,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                strides=2,
                reuse=reuse,
                activation=tf.nn.relu, name='conv5')
            x = tf.layers.conv2d(
                inputs=x,
                filters=512,
                kernel_size=[3, 3],
                padding="same",
                strides=1,
                reuse=reuse,
                activation=tf.nn.relu, name='conv5_1')
            x = tf.layers.conv2d(
                inputs=x,
                filters=1024,
                kernel_size=[3, 3],
                padding="same",
                reuse=reuse,
                strides=2, name='conv6')

        return x


# Train Model
def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    for i in not_initialized_vars:  # only for testing
        print(i.name)

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def train_model(data_loader, sess, merged, loss_op, train_op, input_data, labels_, i, test_writer, train_writer):
    logger.debug('Current epoch : %d' % data_loader.current_epoch)
    logger.debug('step : %d' % i)
    if i % 10 == 0:  # Record summaries and test-set accuracy
        batch_x, batch_y = data_loader.get_next_batch()
        summary, acc = sess.run(
            [merged, loss_op], feed_dict={input_data: batch_x, labels_: batch_y})
        test_writer.add_summary(summary, i)
        logger.debug('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
        batch_x, batch_y = data_loader.get_next_batch()
        summary, _ = sess.run(
            [merged, train_op], feed_dict={input_data: batch_x, labels_: batch_y})
        train_writer.add_summary(summary, i)
        train_loss = sess.run(loss_op,
                              feed_dict={input_data: batch_x, labels_: batch_y})
        logger.debug('Train_error at step %s: %s' % (i, train_loss))


def train(args, datapath, epoches, trajectory_length):
    # configuration
    data_loader = VisualOdometryDataLoader(args, datapath, trajectory_length)
    if USE_CUDA:
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        
        #custom_op.parameter_map["use_off_line"].b = True
        #custom_op.parameter_map["profiling_mode"].b = True
       #mixjingdu
        #custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
       #force_fp32: 
       #custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp32")
       # use tool overflow
       #print("#" * 50)
        #config = npu_tf_config.session_dump_config(config, action='overflow')
        print("#" * 50)

        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  #   must close
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # also
        sess = tf.Session(config=config) 
    else:
        config_proto = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=npu_config_proto(config_proto=config_proto))
    pose_size = 6

    # only for gray scale dataset, for colored channels will be 6
    height, width, channels = 384, 1280, 6

    with tf.name_scope('input'):#定义命名空间
        # placeholder for input
        input_data = tf.placeholder(tf.float32, [args.bsize, args.time_steps, height, width, channels])
        # placeholder for labels
        labels_ = tf.placeholder(tf.float32, [args.bsize, args.time_steps, pose_size])

    with tf.name_scope('unstacked_input'):
        # Unstacking the input into list of time series
        data = tf.unstack(input_data, args.time_steps, 1)
        # Unstacking the labels into the time series
        pose_labels = tf.unstack(labels_, args.time_steps, 1)

    # Building the RCNN Network which
    # which returns the time series of output layers
    with tf.name_scope('RCNN'):
        model = DeepVONet(args, data)
        (outputs, _) = (model.outputs, model.state)
    ## Output layer to compute the output
    with tf.name_scope('weights'):
        regression_w = tf.get_variable('regression_w', shape=[1440, pose_size], dtype=tf.float32)
    with tf.name_scope('biases'):
        regression_b = tf.get_variable("regression_b", shape=[pose_size], dtype=tf.float32)

    # Pose estimate by multiplication with RCNN_output and Output layer
    with tf.name_scope('Wx_plus_b'):
        sess = tf.Session()
        # with tf.Session():
        #       print(outputs[3].dtype)#added
        # print("2")

        if isinstance([outputs],list):
            print("outputs is list")
        # pose_estimated = [tf.nn.xw_plus_b([tf.reshape(output_state, [1, 1440])], [regression_w],
        #                                   [regression_b]) for output_state in outputs]
        pose_estimated = [tf.nn.xw_plus_b(outputs, regression_w, regression_b)]
        max_time = len(pose_estimated)

    # Converting the list of tensor into a tensor
    # Probably this is the part that is unnecessary and causing problems (slowing down the computations)

    # Loss function for all the frames in a batch
    with tf.name_scope('loss_l2_norm'):
        position = [pose_es[:, :3] - pose_lab[:, :3] for pose_es, pose_lab in zip(pose_estimated, pose_labels)]
        angles = [pose_es[:, 3:6] - pose_lab[:, 3:6] for pose_es, pose_lab in zip(pose_estimated, pose_labels)]
        pose_error = (tf.square(position))
        angle_error = (tf.square(angles))
        loss_op = tf.reduce_sum(pose_error + K * angle_error, name='loss')
        tf.summary.scalar('loss_l2_norm', loss_op)

    # optimizer
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr,
                                           beta1=0.9,
                                           beta2=0.999,
                                           epsilon=1e-08,
                                           use_locking=False,
                                           name='Adam')

        train_op = optimizer.minimize(loss_op)
        #opt = optimizer
        #dongtai
        #loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000,decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        #jingtai
        #loss_scale_manager = FixedLossScaleManager(loss_scale=1)
        
        #loss_scale_optimizer = NPULossScaleOptimizer(opt, loss_scale_manager)
        #train_op = loss_scale_optimizer.minimize(loss_op)
    #以下为导入模型继续训练
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('/home/TestUser06/convlstm/npu1_4/outputs/GPUoutput6/'))
    print("restore  success!!!!!")
    # Merge all the summeries and write them out to args.datapath
    # by default ./args.datapath
    merged = tf.summary.merge_all()

    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

    saver = tf.train.Saver()
    initialize_uninitialized(sess)
    train_writer = tf.summary.FileWriter(args.outputpath + 'train', sess.graph)
    test_writer = tf.summary.FileWriter(args.outputpath + 'test')

    i = 0
    while data_loader.current_epoch < epoches:
        start = datetime.datetime.now()
        train_model(data_loader, sess, merged, loss_op, train_op, input_data, labels_, i, test_writer, train_writer)
        i += 1
        end = datetime.datetime.now()
        logger.debug('time at step %s: %s' % (i-1, end - start))
        
        #lossScale = tf.get_default_graph().get_tensor_by_name("train/loss_scale:0")
        #l_s = sess.run(lossScale)
        #print("#" * 100)
        #logger.debug('loss_scale at step %s is : %s'%(i-1, l_s)) 
        #print("#" * 100)
        if data_loader.current_epoch % 50 == 0:#added
            save_path = saver.save(sess, args.outputpath + 'model' + str(data_loader.current_epoch))#added
    save_path = saver.save(sess, args.outputpath + 'model')
    print("Model saved in file: %s" % save_path)
    print("epochs trained: " + str(data_loader.current_epoch))
    train_writer.close()
    test_writer.close()


# Dataset
def default_image_loader(path):
    img = cv2.imread(path)
    if img is not None:
        # Normalizing and Subtracting mean intensity value of the corresponding image
        img = img / np.max(img)
        img = img - np.mean(img)
        img = cv2.resize(img, (1280, 384), fx=0, fy=0)
    return img


class VisualOdometryDataLoader(object):
    def __init__(self, args, datapath, trajectory_length, loader=default_image_loader):
        self.args = args
        self._current_initial_frame = 0
        self._current_trajectory_index = 0
        self.current_epoch = 0

        self.sequences = [4]

        self.base_path = datapath
        self.poses = self.load_poses()
        self.trajectory_length = len(self.sequences)
        self.loader = loader

    def get_image(self, sequence, index):
        image_path = os.path.join(self.base_path, 'sequences', '%02d' % sequence, 'image_0', '%06d' % index + '.png')
        image = self.loader(image_path)
        return image

    def load_poses(self):
        all_poses = []
        for sequence in self.sequences:
            with open(os.path.join(self.base_path, 'poses/', ('%02d' % sequence) + '.txt')) as f:
                poses = np.array([[float(x) for x in line.split()] for line in f], dtype=np.float32)
                all_poses.append(poses)
        return all_poses

    def _set_next_trajectory(self):
        if (self._current_trajectory_index < self.trajectory_length-1):
            self._current_trajectory_index += 1
        else:
            self.current_epoch += 1
            self._current_trajectory_index = 0

        self._current_initial_frame = 0

    def get_next_batch(self):
        img_batch = []
        label_batch = []

        poses = self.poses[self._current_trajectory_index]

        for j in range(self.args.bsize):
            img_stacked_series = []
            labels_series = []

            read_img = self.get_image(self.sequences[self._current_trajectory_index],
                                      self._current_initial_frame + self.args.time_steps)
            if (read_img is None): self._set_next_trajectory()

            for i in range(self._current_initial_frame, self._current_initial_frame + self.args.time_steps):
                img1 = self.get_image(self.sequences[self._current_trajectory_index], i)
                img2 = self.get_image(self.sequences[self._current_trajectory_index], i + 1)
                # print(self.sequences[self._current_trajectory_index])
                # print(img1.size())
                # print(img2.size())
                img_aug = np.concatenate([img1, img2], -1)
                img_stacked_series.append(img_aug)
                # pose = self.get6DoFPose(poses[i, :]) - self.get6DoFPose(poses[self._current_initial_frame, :])#changed
                pose = self.get6DoFPose(poses[i + 1, :]) - self.get6DoFPose(poses[i, :])
                labels_series.append(pose)
            img_batch.append(img_stacked_series)
            label_batch.append(labels_series)
            self._current_initial_frame += self.args.time_steps
        label_batch = np.array(label_batch)
        img_batch = np.array(img_batch)
        return img_batch, label_batch

    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self, R):
        assert (self.isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def get6DoFPose(self, p):
        pos = np.array([p[3], p[7], p[11]])
        R = np.array([[p[0], p[1], p[2]], [p[4], p[5], p[6]], [p[8], p[9], p[10]]])
        angles = self.rotationMatrixToEulerAngles(R)
        return np.concatenate((pos, angles))

# Main_Args Class
class MyArgs():
    def __init__(self, datapath, outputpath, bsize, trajectory_length, lr=0.001, time_steps=100, train_iter=5):
        self.datapath = datapath
        self.outputpath = outputpath
        self.bsize = bsize
        self.trajectory_length = trajectory_length
        self.lr = lr
        self.time_steps = time_steps
        self.train_iter = train_iter

"""以下为测试模块"""
args = MyArgs(datapath='/home/TestUser06/convlstm/dataset/',
              outputpath='/home/TestUser06/convlstm/npu1_4/outputs/output1/',
              bsize=32,
              trajectory_length=4,
              train_iter=1,
              time_steps=TIME_STEPS)

# configuration
config_proto = tf.ConfigProto(device_count = {'GPU': 0})
sess = tf.Session(config=config_proto)
data_loader = VisualOdometryDataLoader(args,args.datapath,args.trajectory_length)
""" input_batch must be in shape of [?, TIME_STEPS, 384, 1280, 6] """
#tf.reset_default_graph()
#print('Restoring Entire Session from checkpoint : %s'%args.outputpath+"model300.meta")
imported_meta = tf.train.import_meta_graph(args.outputpath + "model300.meta")
print('Success')
imported_meta.restore(sess, tf.train.latest_checkpoint(args.outputpath))#查找最新保存的ckpt文件名，并加载,latest_filename=model240
input_data = tf.get_default_graph().get_tensor_by_name("input/Placeholder:0")#以下5句是加载变量的过程
# placeholder for labels 下两句的作用？？
labels_ = tf.get_default_graph().get_tensor_by_name("input/Placeholder_1:0")
loss_op = tf.get_default_graph().get_tensor_by_name("loss_l2_norm/loss:0")
poses = []
poses.append(tf.get_default_graph().get_tensor_by_name("Wx_plus_b/xw_plus_b:0"))
# poses.append(tf.get_default_graph().get_tensor_by_name("Wx_plus_b/xw_plus_b_1:0"))
# poses.append(tf.get_default_graph().get_tensor_by_name("Wx_plus_b/xw_plus_b_2:0"))
# poses.append(tf.get_default_graph().get_tensor_by_name("Wx_plus_b/xw_plus_b_3:0"))
# poses.append(tf.get_default_graph().get_tensor_by_name("Wx_plus_b/xw_plus_b_4:0"))
while data_loader.current_epoch < args.train_iter:
    input_, ground_truth_batch = data_loader.get_next_batch()#图片，位姿差真值
    #print("shape(ground_truth_batch)",ground_truth_batch.shape)
    output = sess.run(poses, feed_dict={input_data:input_})#位姿差预估值
    #print("shape(output)",np.array(output).shape)
    print('Current epoch : %d' % data_loader.current_epoch)
    print('output length : %d' % len(output))
    for i in range(len(output)):
       for j in range(32):
            fh = open("/home/TestUser06/convlstm/npu1_4/txtcsv/output_file.txt", "a")#位姿真值存放文件
            fh.write("%f %f %f %f %f %f\n"%(ground_truth_batch[j,i,0],
                                            ground_truth_batch[j,i,1],
                                            ground_truth_batch[j,i,2],
                                            ground_truth_batch[j,i,3],
                                            ground_truth_batch[j,i,4],
                                            ground_truth_batch[j,i,5]))
            fh.close()
            fh = open("/home/TestUser06/convlstm/npu1_4/txtcsv/estimated.txt","a")#位姿预估值存放文件
            fh.write("%f %f %f %f %f %f\n"%(output[i][j,0],
                                            output[i][j,1],
                                            output[i][j,2],
                                            output[i][j,3],
                                            output[i][j,4],
                                            output[i][j,5]))
            fh.close()
file_old = open("/home/TestUser06/convlstm/npu1_4/txtcsv/output_file.txt", 'rb+')
lines = file_old.readlines()
# 定位到最后一行的行首，若要删除后N行，将lines[-1]改为lines[-N:]即可
file_old.seek(-len(lines[-1]), os.SEEK_END)
file_old.truncate()  # 截断之后的数据
file_old.close()

file_oldd = open("/home/TestUser06/convlstm/npu1_4/txtcsv/estimated.txt", 'rb+')
liness = file_oldd.readlines()
# 定位到最后一行的行首，若要删除后N行，将lines[-1]改为lines[-N:]即可
file_oldd.seek(-len(liness[-1]), os.SEEK_END)
file_oldd.truncate()  # 截断之后的数据
file_oldd.close()
