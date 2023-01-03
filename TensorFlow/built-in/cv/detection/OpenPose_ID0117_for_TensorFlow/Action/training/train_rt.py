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
from npu_bridge.npu_init import *
import pandas as pd
from enum import Enum
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model

import matplotlib.pyplot as plt
from keras.callbacks import Callback
import itertools
from sklearn.metrics import confusion_matrix
#*****npu modify begin*****
import tensorflow as tf
from keras import backend as K
from npu_bridge.npu_init import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', dest='data_path', default='./data/', help='path of the dataset')
parser.add_argument('--precision_mode', dest='precision_mode', default='allow_mix_precision', help='precision mode')
parser.add_argument('--over_dump', dest='over_dump', default='False', help='if or not over detection')
parser.add_argument('--over_dump_path', dest='over_dump_path', default='./overdump', help='over dump path')
parser.add_argument('--data_dump_flag', dest='data_dump_flag', default='False', help='data dump flag')
parser.add_argument('--data_dump_step', dest='data_dump_step', default='10', help='data dump step')
parser.add_argument('--data_dump_path', dest='data_dump_path', default='./datadump', help='data dump path')
parser.add_argument('--profiling', dest='profiling', default='False', help='if or not profiling for performance debug')
parser.add_argument('--profiling_dump_path', dest='profiling_dump_path', default='./profiling', help='profiling path')
parser.add_argument('--autotune', dest='autotune', default='False', help='whether to enable autotune, default is False')

parser.add_argument('--train_epoch', dest='train_epoch', type=int, default=2000, help='# of step for training')
parser.add_argument('--modeldir', dest='modeldir', default='./ckpt', help='ckpt dir')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='# images in batch')

parser.add_argument("--dynamic_input", type=str, default='1', help="--dynamic_input=1 Use fuzzy compilation. --dynamic_input=lazy_recompile Compile using lazy static graph")
args = parser.parse_args()


def npu_keras_optimizer(opt):
    npu_opt = KerasDistributeOptimizer(opt)
    return npu_opt
#*****npu modify end*****

class Actions(Enum):
    # framewise_recognition.h5
    # squat = 0
    # stand = 1
    # walk = 2
    # wave = 3

    # framewise_recognition_under_scene.h5
    stand = 0
    walk = 1
    operate = 2
    fall_down = 3
    # run = 4


# Callback class to visialize training progress
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# load data
datapath = ("%s/data_with_scene.csv" %(args.data_path))
raw_data = pd.read_csv(datapath, header=0)
dataset = raw_data.values
# X = dataset[:, 0:36].astype(float)
# Y = dataset[:, 36]
X = dataset[0:3289, 0:36].astype(float)  # 忽略run数据
Y = dataset[0:3289, 36]

# 将类别编码为数字
# encoder = LabelEncoder()
# encoder_Y = encoder.fit_transform(Y)
# print(encoder_Y[0], encoder_Y[900], encoder_Y[1800], encoder_Y[2700])
# encoder_Y = [0]*744 + [1]*722 + [2]*815 + [3]*1008 + [4]*811
encoder_Y = [0]*744 + [1]*722 + [2]*815 + [3]*1008
# one hot 编码
dummy_Y = np_utils.to_categorical(encoder_Y)

# train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.1, random_state=9)

# build keras model
model = Sequential()
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=4, activation='softmax'))  # units = nums of classes

# training
#*****npu modify begin*****
sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["jit_compile"].b = True
#custom_op.parameter_map["dynamic_input"].b = True
#if args.dynamic_input == "lazy_recompile":
#    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
#elif args.dynamic_input == "1":
#    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("dynamic_execute")
#else:
#    print("Enter correct compilation parameters.")
#custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)
if args.data_dump_flag.strip() == "True":
    custom_op.parameter_map["enable_dump"].b = True
    custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args.data_dump_path)
    custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(args.data_dump_step)
    custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
if args.over_dump.strip() == "True":
    # dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限
    custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args.over_dump_path)
    # enable_dump_debug：是否开启溢出检测功能
    custom_op.parameter_map["enable_dump_debug"].b = True
    # dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
    custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
if args.profiling.strip() == "True":
    custom_op.parameter_map["profiling_mode"].b = False
    profilingvalue = (
            '{"output":"%s","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":""}' % (
        args.profiling_dump_path))
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(profilingvalue)
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)
#*****npu modify end*****

his = LossHistory()
model.compile(loss='categorical_crossentropy', optimizer=Adam(args.learning_rate), metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=args.batch_size, epochs=args.train_epoch, verbose=1, validation_data=(X_test, Y_test), callbacks=[his])
model.summary()
his.loss_plot('epoch')

#*****npu modify begin*****
print('====save model====')
os.makedirs(args.modeldir, exist_ok=True)
ckptparams = ("%s/model_weights.h5" %(args.modeldir))
ckptall = ("%s/model.h5" %(args.modeldir))
model.save_weights(ckptparams)
model.save(ckptall)
sess.close()
#*****npu modify end*****

# model.save('framewise_recognition.h5')

# # evaluate and draw confusion matrix
# print('Test:')
# score, accuracy = model.evaluate(X_test,Y_test,batch_size=32)
# print('Test Score:{:.3}'.format(score))
# print('Test accuracy:{:.3}'.format(accuracy))
# # confusion matrix
# Y_pred = model.predict(X_test)
# cfm = confusion_matrix(np.argmax(Y_test,axis=1), np.argmax(Y_pred, axis=1))
# np.set_printoptions(precision=2)
#
# plt.figure()
# class_names = ['squat', 'stand', 'walk', 'wave']
# plot_confusion_matrix(cfm, classes=class_names, title='Confusion Matrix')
# plt.show()

# # test
# model = load_model('framewise_recognition.h5')
#
# test_input = [0.43, 0.46, 0.43, 0.52, 0.4, 0.52, 0.39, 0.61, 0.4,
#               0.67, 0.46, 0.52, 0.46, 0.61, 0.46, 0.67, 0.42, 0.67,
#               0.42, 0.81, 0.43, 0.91, 0.45, 0.67, 0.45, 0.81, 0.45,
#               0.91, 0.42, 0.44, 0.43, 0.44, 0.42, 0.46, 0.44, 0.46]
# test_np = np.array(test_input)
# test_np = test_np.reshape(-1, 36)
#
# test_np = np.array(X[1033]).reshape(-1, 36)
# if test_np.size > 0:
#     pred = np.argmax(model.predict(test_np))
#     init_label = Actions(pred).name
#     print(init_label)
