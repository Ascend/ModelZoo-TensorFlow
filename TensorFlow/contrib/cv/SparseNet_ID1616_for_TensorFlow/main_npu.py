#!/usr/bin/env python
# -*- coding: UTF-8 -*-
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
import tensorflow.compat.v1 as tf

import argparse
#import moxing as mox
from datetime import datetime
import os
from datetime import datetime

os.system('pip uninstall -y enum34')
os.system('pip install tensorpack')

os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = "0"

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils import logger

from tensorpack.tfutils.sesscreate import NewSessionCreator
from npu_bridge.npu_init import *

"""
CIFAR10 DenseNet example. See: http://arxiv.org/abs/1608.06993
Code is developed based on Yuxin Wu's ResNet implementation: https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet
Results using DenseNet (L=40, K=12) on Cifar10 with data augmentation: ~5.77% test error.

Running time:
On one TITAN X GPU (CUDA 7.5 and cudnn 5.1), the code should run ~5iters/s on a batch size 64.
"""


class MyModelSaver(ModelSaver):
    """
    rewrite ModelSaver for graph.pbtxt.
    """

    def _before_train(self):
        # graph is finalized, OK to write it now.
        self.saver.export_meta_graph(
            os.path.join(self.checkpoint_dir, 'graph.pbtxt'),
            as_text=True,
            collection_list=self.graph.get_all_collection_keys(),
        )


def linearFetch(lst):
    return lst


def expFetch(lst):
    nums = len(lst)
    i = 1
    res = []
    while i <= nums:
        res.append(lst[nums - i])
        i *= 2
    # res.append(lst[-1])
    return res


def conv3x3(name, input_data, out_channels, stride=1):
    return Conv2D(name, input_data, out_channels, 3, stride=stride, nl=tf.identity, use_bias=False,
                  W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / out_channels)))


def conv1x1(name, input_data, out_channels, stride=1):
    return Conv2D(name, input_data, out_channels, 1, stride=stride, nl=tf.identity, use_bias=False,
                  W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / out_channels)))


def add_layer_without_concat(name, l, growth_rate):
    # basic BN-ReLU-Conv unit
    with tf.variable_scope(name) as scope:
        c = BatchNorm('bn1', l)
        c = tf.nn.relu(c)
        c = conv3x3('conv1', c, growth_rate, 1)
    return c


def add_layer(name, l, growth_rate):
    c = add_layer_without_concat(name, l, growth_rate)
    return tf.concat([c, l], 3)


def add_transition(name, l):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        l = BatchNorm('bn1', l)
        l = tf.nn.relu(l)
        l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
        l = AvgPooling('pool', l, 2)
    return l


class Model(ModelDesc):
    def __init__(self, depth, growth_rate=12, fetch="dense",
                 bottleneck=False, compression=1, dropout=0,
                 num_classes=10):

        super(Model, self).__init__()
        # self.N = int((depth - 4) / 3)
        self.N = depth // 3
        if isinstance(growth_rate, list):
            self.growthRate1 = growth_rate[0]
            self.growthRate2 = growth_rate[1]
            self.growthRate3 = growth_rate[2]
        elif isinstance(growth_rate, int):
            self.growthRate1 = growth_rate
            self.growthRate2 = growth_rate
            self.growthRate3 = growth_rate
        else:
            raise ValueError("growth_rate:[%s] ERROR" % str(growth_rate))
        self.fetch = fetch
        self.bottleneck = bottleneck
        self.compression = compression
        self.num_classes = num_classes

    def inputs(self):
        # return [tf.TensorSpec([None, 32, 32, 3], tf.float32, 'input'),
        #         tf.TensorSpec([None], tf.int32, 'label')]
        return [
            tf.placeholder(name='input', shape=[None, 32, 32, 3], dtype=tf.float32),
            tf.placeholder(name='label', shape=[None], dtype=tf.int32),
        ]
        # return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
        #         InputDesc(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        # image, label = input_vars
        image = image / 255.  # convert range to 0 ~ 1

        def dense_net(name, num_classes=10):
            l = conv3x3('conv0', image, 16, 1)

            if self.fetch == "dense":
                fetch = linearFetch
            else:
                fetch = expFetch

            with tf.variable_scope('block1') as scope:
                saved_activations = [l]
                for i in range(self.N):
                    l = add_layer_without_concat('densen_layer.{}'.format(i), l, self.growthRate1)
                    saved_activations.append(l)

                    l = tf.concat(fetch(saved_activations), 3)
                l = add_transition('transition1', l)

            with tf.variable_scope('block2') as scope:
                saved_activations = [l]
                for i in range(self.N):
                    l = add_layer_without_concat('densen_layer.{}'.format(i), l, self.growthRate2)
                    saved_activations.append(l)
                    l = tf.concat(fetch(saved_activations), 3)
                l = add_transition('transition2', l)

            with tf.variable_scope('block3') as scope:
                saved_activations = [l]
                for i in range(self.N):
                    l = add_layer_without_concat('densen_layer.{}'.format(i), l, self.growthRate3)
                    saved_activations.append(l)
                    l = tf.concat(fetch(saved_activations), 3)

            l = BatchNorm('bn_last', l)
            l = tf.nn.relu(l)
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, out_dim=self.num_classes, nl=tf.identity)

            return logits

        logits = dense_net("dense_net")

        tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='incorrect_vector')
        # wrong = prediction_incorrect(logits, label)
        #
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W
        # wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
        #                                   480000, 0.2, True)
        # wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        wd_cost = tf.multiply(1e-4, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))  # monitor W

        return tf.add_n([cost, wd_cost], name='cost')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        # return tf.train.AdamOptimizer()
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test, args):
    isTrain = train_or_test == 'train'
    if args.dataset == "c10":
        dst = dataset.Cifar10
        dir = args.dataset_dir
    elif args.dataset == "c100":
        dst = dataset.Cifar100
        dir = args.dataset_dir

    ds = dst(train_or_test, dir=dir, shuffle=True)

    pp_mean = ds.get_per_pixel_mean()
    # pc_mean = np.array([125.3, 123.0, 113.9])
    # pc_std = np.array([63.3, 62.1, 66.7])
    if isTrain:
        augmentors = [
            # imgaug.MapImage(lambda x: (x - pc_mean) / pc_std),
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            # imgaug.Brightness(20),
            # imgaug.Contrast((0.6,1.4)),
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    else:
        augmentors = [
            # imgaug.MapImage(lambda x: (x - pc_mean) / pc_std),
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    # 当前非整batch需要丢弃
    # ds = BatchData(ds, args.batch_size, remainder=not isTrain)
    ds = BatchData(ds, args.batch_size, remainder=isTrain)
    # if isTrain:
    #     ds = PrefetchData(ds, 3, 2)  # window can not use it with lambda!
    return ds


def get_config(args):
    # log_dir = 'train_log/%s-%d-%d-%s-single-fisrt%s-second%s-max%s' % (
    log_dir = './result/%s/' \
              '%s-' \
              'depth_%d-' \
              'k_%d-' \
              '%s-' \
              'single-' \
              'first%s-second%s-max%s-%s' % (
                  args.log_dir,
                  str(args.fetch),
                  args.depth,
                  args.growth_rate,
                  args.dataset,
                  str(args.drop_1), str(args.drop_2), str(args.max_epoch),
                  datetime.now().strftime('%m%d-%H%M%S'))
    os.makedirs(log_dir)
    # logger.set_logger_dir(log_dir, action='n')

    # prepare dataset
    dataset_train = get_data('train', args)
    steps_per_epoch = dataset_train.size()
    dataset_test = get_data('test', args)

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
    sess_creator = NewSessionCreator(config=sess_config)

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[MyModelSaver(checkpoint_dir=log_dir),
                   InferenceRunner(dataset_test,
                                   [ScalarStats('cost'), ClassificationError('incorrect_vector')]),
                   ScheduledHyperParamSetter('learning_rate',
                                             [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001)])],
        model=Model(
            depth=args.depth, growth_rate=args.growth_rate, fetch=args.fetch, num_classes=args.num_classes),
        steps_per_epoch=steps_per_epoch,
        max_epoch=args.max_epoch,
        session_creator=sess_creator
    )


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument('-g', '--gpu', default="0",
                        help='comma separated list of GPU(s) to use.')  # nargs='*' in multi mode
    parser.add_argument('--load', default=False, help='is load model from checkpoint?')
    parser.add_argument('--log_dir', default="train_log", help="The root directory to save training logs.")
    parser.add_argument('--dataset', default="c10", type=str, choices=["c10", "c100"])
    parser.add_argument('--dataset_dir', default="/cache/dataset/cifar10_data", type=str)
    parser.add_argument('--name', default=None)

    # model related
    # parser.add_argument('--arch')
    parser.add_argument('--fetch', default="sparse", type=str, choices=["dense", "sparse"])
    parser.add_argument('-d', '--depth', default=100, type=int, help='The depth of densenet')
    parser.add_argument('-gr', '--growth_rate', default=24, type=int,
                        help='other name:k, The number of output filters ')
    parser.add_argument('--growth-step', default=None)

    parser.add_argument('--bottleneck', default=0, type=int, help="Whether to use bottleneck")
    parser.add_argument('--compression', default=0, type=float, help="Whether to use compression")
    parser.add_argument('--dropout', default=0.0, type=float, help="The ratio of dropout layer")

    # optimizer
    parser.add_argument('--batch_size', default=64, type=int, help="Batch fed into graph every iter")
    parser.add_argument('--drop_1', default=150, help='Epoch to drop learning rate to 0.01.')  # nargs='*' in multi mode
    parser.add_argument('--drop_2', default=225, help='Epoch to drop learning rate to 0.001')
    parser.add_argument('--max_epoch', default=300, help='max epoch')

    # modelarts
    parser.add_argument("--data_url", type=str, default="./dataset")
    parser.add_argument("--train_url", type=str, default="./output")
    parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()

    # 在ModelArts容器创建数据存放目录
    print("start copy data ===>>> modelarts")
    print("args.data_url:[%s]" % args.data_url)
    print("args.train_url:[%s]" % args.train_url)
    #temp_data_dir = "/cache/dataset"
   # os.makedirs(temp_data_dir)
    # OBS数据拷贝到ModelArts容器内
    #mox.file.copy_parallel(args.data_url, temp_data_dir)
    #print("over copy data ===>>> modelarts")

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    BATCH_SIZE = args.batch_size

    nr_tower = 0

    if args.gpu:
        nr_tower = len(args.gpu.split(','))
        # nr_tower = [1]
        BATCH_SIZE = BATCH_SIZE // nr_tower

    if args.dataset == 'c10':
        args.num_classes = 10
    elif args.dataset == 'c100':
        args.num_classes = 100
    else:
        raise NotImplementedError

    config = get_config(args)
    if args.load:
        config.session_init = SaverRestore(args.load)

    #print('/cache/:' + str(os.listdir('/cache/')))

    # SyncMultiGPUTrainer(config).train()
    # launch_train_with_config(config, SyncMultiGPUTrainer(nr_tower))
    trainer = SimpleTrainer()
    launch_train_with_config(config, trainer)

    # ModelArts容器训练输出目录
    #temp_result_dir = "/cache/%s/" % args.log_dir
    # os.makedirs(temp_result_dir)
   # print('/cache/:' + str(os.listdir('/cache/')))

    # 在ModelArts容器创建训练输出目录
   # print("start copy result to obs")
    #mox.file.copy_parallel(temp_result_dir, args.train_url)
    #print("over copy result to obs")