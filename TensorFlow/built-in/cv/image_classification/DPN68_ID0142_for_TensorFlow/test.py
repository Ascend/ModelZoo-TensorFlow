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
# ==============================================================================

from npu_bridge.npu_init import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import DPN as net
import numpy as np
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import math
import time
import argparse

# set log level
# 0 = INFO + WARNING + ERROR + FATAL
# 1 = WARNING + ERROR + FATAL
# 2 = ERROR + FATAL
# 3 = FATAL
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Begin: Found GPU Device")
begin = time.time()
# GPU device
device_name = tf.test.gpu_device_name()
if device_name is not '':
    print('Found GPU Device!')
else:
    print('Found GPU Device Failed!')
cost = time.time() - begin
print("Found GPU Device cost {} sec !".format(cost))

print("Begin: config")
begin = time.time()
# set config
config = {
    # dpn92 [96, 96, 256], dpb98 [160, 160, 256]
    'first_dpn_block_filters': [96, 96, 256],
    # dpn92 [3, 4, 20, 3], dpb98 [3, 6, 20, 3]
    'dpn_block_list': [3, 4, 20, 3],

    # parameters for conv and pool before dense block
    'init_conv_filters': [16],
    'init_conv_kernel_size': [3],
    'init_conv_strides': [1],
    'init_pooling_pool_size': 3,
    'init_pooling_strides': 2,

    # dpn92 [16, 32, 24, 128], dpb98 [16, 32, 32, 128]
    'k': [16, 32, 24, 128],
    # dpn92 32, dpb98 40
    'G': 32
}
cost = time.time() - begin
print("config cost {} sec !".format(cost))

print("Begin: params")
begin = time.time()
# set params
parser = argparse.ArgumentParser(description="train params")
parser.add_argument("--num_train", type=int, default=640, help='num_train.')
parser.add_argument("--train_steps", type=int, default=100, help='train_steps.')
parser.add_argument("--train_batch_size", type=int, default=128, help='train_batch_size.')
parser.add_argument("--train_epochs", type=int, default=1, help='train_epochs.')
parser.add_argument("--num_classes", type=int, default=10, help='num_classes.')
parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight_decay.')
parser.add_argument("--keep_prob", type=float, default=0.8, help='keep_prob.')
parser.add_argument("--output_path", type=str, default="output", help='output_path.')
parser.add_argument("--num_test", type=int, default=10000, help='num_test.')
parser.add_argument("--test_steps", type=int, default=100, help='test_steps.')
parser.add_argument("--test_batch_size", type=int, default=200, help='test_batch_size.')
args = parser.parse_args()
mean = np.array([123.68, 116.779, 103.979]).reshape((1, 1, 1, 3))
data_shape = (32, 32, 3)
lr = math.sqrt(0.1)
args.num_train = args.train_steps * args.train_batch_size
args.num_test = args.test_steps * args.test_batch_size

cost = time.time() - begin
print("params cost {} sec !".format(cost))

print("Begin: load_data")
begin = time.time()
(x_train, y_train) , (x_test, y_test) = cifar10.load_data()
cost = time.time() - begin
print("load_data cost {} sec !".format(cost))

print("Begin: to_categorical")
begin = time.time()
y_train = keras.utils.to_categorical(y_train, args.num_classes)
y_test = keras.utils.to_categorical(y_test, args.num_classes)
cost = time.time() - begin
print("to_categorical cost {} sec !".format(cost))

print("Begin: ImageDataGenerator")
begin = time.time()
train_gen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1).flow(x_train, y_train, batch_size=int(args.train_batch_size))
test_gen = ImageDataGenerator().flow(x_test, y_test, batch_size=args.test_batch_size)
cost = time.time() - begin
print("ImageDataGenerator cost {} sec !".format(cost))

reduce_lr_epoch = [(int(args.train_epochs))//2, 3*(int(args.train_epochs))//4]

print("Begin: net.DPN")
begin = time.time()
testnet = net.DPN(config, data_shape, args.num_classes, args.weight_decay, args.keep_prob, 'channels_last')
cost = time.time() - begin
print("net.DPN cost {} sec !".format(cost))

print("Begin: write_graph")
begin = time.time()
tf.io.write_graph(testnet.sess.graph, args.output_path, 'graph.pbtxt', as_text=True)
cost = time.time() - begin
print("write_graph cost {} sec !".format(cost))

rank_size = int(os.getenv('RANK_SIZE'))
rank_id = int(os.getenv('RANK_ID'))

for epoch in range(int(args.train_epochs)):

    print('-'*20, 'epoch', epoch, '-'*20)
    train_acc = []
    train_loss = []
    test_acc = []
    # reduce learning rate
    if epoch in reduce_lr_epoch:
        lr = lr * 0.1 *rank_size
        print('reduce learning rate =', lr, 'now')
    # train one epoch
    begin_epoch = time.time()
    for iter in range(int(args.num_train)//int(args.train_batch_size)):
        # get and preprocess image
        images1, labels1 = train_gen.next()
        images = images1[rank_id * args.train_batch_size / rank_size:(rank_id + 1) * args.train_batch_size / rank_size]
        if images.shape[0] == 0:
            break
        labels = labels1[rank_id * args.train_batch_size / rank_size:(rank_id + 1) * args.train_batch_size / rank_size]
        images = images - mean
        # train_one_batch also can accept your own session
        begin = time.time()
        loss, acc = testnet.train_one_batch(images, labels, lr)
        end = time.time()
        train_acc.append(acc)
        train_loss.append(loss)
        print('step:', str(iter + 1) + '/' +str(int(args.num_train) // int(args.train_batch_size)), 'time(ms):', '%.3f' % ((end - begin) * 1000), 'loss:', loss, 'acc:', acc)
        sys.stdout.flush()
    end_epoch = time.time()
    mean_train_loss = np.mean(train_loss)
    mean_train_acc = np.mean(train_acc)
    print('>> epoch:', str(epoch + 1), 'total_time(ms):', '%.3f' % ((end_epoch - begin_epoch) * 1000), 'avg_loss:', mean_train_loss, 'avg_acc:', mean_train_acc)
    sys.stdout.flush()
    # save the lastest ckpt
    if int(epoch + 1) == int(args.train_epochs):
        tf.train.Saver().save(testnet.sess, args.output_path+"/model.ckpt-" + str(epoch))

    # validate one epoch
    #for iter in range(args.num_test//args.test_batch_size):
        # get and preprocess image
        #images, labels = test_gen.next()
        #images = images - mean
        # validate_one_batch also can accept your own session
        #logit, acc = testnet.validate_one_batch(images, labels)
        #test_acc.append(acc)
        #sys.stdout.write("\r>> test "+str(iter+1)+'/'+str(args.num_test//args.test_batch_size)+' acc '+str(acc))
    #mean_val_acc = np.mean(test_acc)
    #sys.stdout.write("\n")
    #print('>> epoch', epoch, ' test mean acc', mean_val_acc)

    # logit = testnet.test(images)
    # testnet.save_weight(self, mode, path, sess=None)
    # testnet.load_weight(self, mode, path, sess=None)