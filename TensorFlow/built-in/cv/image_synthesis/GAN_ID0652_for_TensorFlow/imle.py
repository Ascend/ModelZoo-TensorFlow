#! -*- coding: utf-8 -*-
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
#《Implicit Maximum Likelihood Estimation》例子

from npu_bridge.npu_init import *
import numpy as np
import scipy as sp
from scipy import misc
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import RMSprop
from keras.callbacks import Callback
import os,json
import warnings
import tensorflow as tf

import argparse
warnings.filterwarnings("ignore") # 忽略keras带来的满屏警告


parser = argparse.ArgumentParser(description="train gan")
parser.add_argument('--epochs', default=10, help='train epochs')
parser.add_argument('--batch_size', default=128, help='train batch size')
parser.add_argument('--data_path', default='data', help="train data path")
parser.add_argument('--precision_mode', default='allow_fp32_to_fp16', help='train precision mode')
parser.add_argument('--loss_scale', default=False, help="use loss scale or not")
parser.add_argument('--over_dump', default='False', help='if or not over detection')
parser.add_argument('--over_dump_path', dest='over_dump_path', default='./overdump', help='over dump path')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
if not os.path.exists('samples'):
    os.mkdir('samples')
###################NPU_modify_start##################
sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["dynamic_input"].b = 1
custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(args.precision_mode)
if(args.over_dump.strip()=="True"):
    print("*************over flow check***************")
    # dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限
    custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(args.over_dump_path)
    # enable_dump_debug：是否开启溢出检测功能
    custom_op.parameter_map["enable_dump_debug"].b = True
    # dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
    custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)
###################NPU_modify_end##################

imgs = glob.glob(args.data_path + '/train/*/*.png')
np.random.shuffle(imgs)
img_dim = 128
z_dim = 128
num_layers = int(np.log2(img_dim)) - 3
max_num_channels = img_dim * 8
f_size = img_dim // 2**(num_layers + 1)
batch_size = int(args.batch_size)


def imread(f, mode='gan'):
    x = misc.imread(f, mode='RGB')
    if mode == 'gan':
        x = misc.imresize(x, (img_dim, img_dim))
        x = x.astype(np.float32)
        return x / 255 * 2 - 1
    elif mode == 'fid':
        x = misc.imresize(x, (299, 299))
        return x.astype(np.float32)


class img_generator:
    """图片迭代器，方便重复调用
    """
    def __init__(self, imgs, mode='gan', batch_size=64):
        self.imgs = imgs
        self.batch_size = batch_size
        self.mode = mode
        if len(imgs) % batch_size == 0:
            self.steps = len(imgs) // batch_size
        else:
            self.steps = len(imgs) // batch_size + 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        X = []
        while True:
            np.random.shuffle(self.imgs)
            for i,f in enumerate(self.imgs):
                X.append(imread(f, self.mode))
                if len(X) == self.batch_size or i == len(self.imgs)-1:
                    X = np.array(X)
                    if self.mode == 'gan':
                        Z = np.random.randn(len(X), z_dim)
                        self.XZ = [X, Z]
                        yield [X, Z], None
                    elif self.mode == 'fid':
                        yield X
                    X = []


# 生成器
z_in = Input(shape=(z_dim, ))
z = z_in

z = Dense(f_size**2 * max_num_channels)(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Reshape((f_size, f_size, max_num_channels))(z)

for i in range(num_layers):
    num_channels = max_num_channels // 2**(i + 1)
    z = Conv2DTranspose(num_channels,
                        (5, 5),
                        strides=(2, 2),
                        padding='same')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

z = Conv2DTranspose(3,
                    (5, 5),
                    strides=(2, 2),
                    padding='same')(z)
z = Activation('tanh')(z)

g_model = Model(z_in, z)
g_model.summary()


# 整合模型
x_in = Input(shape=(img_dim, img_dim, 3))
z_in = Input(shape=(z_dim, ))

x_real = x_in
x_fake = g_model(z_in)

train_model = Model([x_in, z_in], [x_real, x_fake])

x_real = K.expand_dims(x_real, 1)
x_fake = K.expand_dims(x_fake, 0)
loss = K.sum(x_real**2, [2, 3, 4]) + K.sum(x_fake**2, [2, 3, 4]) - 2 * K.sum(x_real * x_fake, [2, 3, 4])
loss = K.mean(K.min(loss, 1))

train_model.add_loss(loss)
########################NPU_modify_start#########################
# optimizer = RMSprop(1e-4)
# if args.loss_scale == True:
#     loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=tf.flags.FLAGS.init_loss_scale_value,
#                                                            incr_every_n_steps=1000, decr_every_n_nan_or_inf=2,
#                                                            decr_ratio=0.8)
#     if int(os.getenv('RANK_SIZE')) == 1:
#         optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager)
#     else:
#         optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager, is_distributed=True)
#
# train_model.compile(optimizer=optimizer, loss='')
########################NPU_modify_end#########################
train_model.compile(optimizer = RMSprop(1e-4))
train_model.summary()


# 采样函数
def sample(path, n=9, z_samples=None):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    if z_samples is None:
        z_samples = np.random.randn(n**2, z_dim)
    for i in range(n):
        for j in range(n):
            z_sample = z_samples[[i * n + j]]
            x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    imageio.imwrite(path, figure)


class Trainer(Callback):
    def __init__(self):
        self.batch = 0
        self.n_size = 9
        self.iters_per_sample = 100
        self.Z = np.random.randn(self.n_size**2, z_dim)
        self.model_id = 0
        self.losses = []
    # *****npu modify begin*****
    def on_batch_end(self, batch, logs=None):
        if self.batch % self.iters_per_sample == 0:
            sample('samples/test_%s.png' % self.batch,
                self.n_size, self.Z)
            #train_model.save_weights(f'./ckpt_npu/train_model_{self.model_id:05}.h5')
            self.model_id += 1
        self.batch += 1
        self.losses.append(logs.get("loss"))
        with open('loss_log.txt', 'a', encoding='utf-8') as f:
            f.write(f'batch:{self.batch:05}  loss:{logs.get("loss"):.4f}\n')
    # *****npu modify end*****

if __name__ == '__main__':

    #npu_keras_sess = set_keras_session_npu_config()

    trainer = Trainer()
    img_data = img_generator(imgs, 'gan', batch_size)
    # *****npu modify begin*****
    ckpt_path = "ckpt_npu"
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    # *****npu modify end*****

    train_model.fit_generator(img_data.__iter__(),
                              steps_per_epoch=len(img_data),
                              epochs=int(args.epochs),
                              callbacks=[trainer])
    #close_session(npu_keras_sess)
    ###################NPU_modify_start##################
    close_session(sess)
    ###################NPU_modify_end##################
