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

from __future__ import absolute_import, division, print_function, unicode_literals
"""
基于tensorflow 高阶API
CGAN  Conditional GAN
损失函数: 基于SGAN的Loss 判别器输出为概率值需要sigmoid
网络结构: MLP 第一层的MLP的要加上 Concat One_Hot 条件
数据形式: 不带卷积 没有深度维  图片压缩到0 1 之间
生成器: sigmoid 映射到0 1 之间 迎合数据格式
判别器: sigmoid 映射到0 1 之间 迎合loss公式的约束
初始化: xavier初始化  即考虑输入输出维度的 glorot uniform
训练： 判别器和生成器同时训练 同步训练 不偏重任一方
"""
import npu_device as npu
import my_mnist
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import time
import argparse
import os
import ast


def npu_config():
  if args.data_dump_flag:
    npu.global_options().dump_config.enable_dump = True
    npu.global_options().dump_config.dump_path = args.data_dump_path
    npu.global_options().dump_config.dump_step = args.data_dump_step
    npu.global_options().dump_config.dump_mode = "all"

  if args.over_dump:
    npu.global_options().dump_config.enable_dump_debug = True
    npu.global_options().dump_config.dump_path = args.over_dump_path
    npu.global_options().dump_config.dump_debug_mode = "all"

  if args.profiling:
    npu.global_options().profiling_config.enable_profiling = True
    profiling_options = '{"output":"' + args.profiling_dump_path + '", \
                        "training_trace":"on", \
                        "task_trace":"on", \
                        "aicpu":"on", \
                        "L2": "on", \
                        "aic_metrics":"PipeUtilization",\
                        "fp_point":"", \
                        "bp_point":""}'
    npu.global_options().profiling_config.profiling_options = profiling_options
  npu.global_options().precision_mode=args.precision_mode
  if args.use_mixlist and args.precision_mode=='allow_mix_precision':
    npu.global_options().modify_mixlist=args.mixlist_file
  if args.fusion_off_flag:
    npu.global_options().fusion_switch_file=args.fusion_off_file
  if args.auto_tune:
    npu.global_options().auto_tune_mode="RL,GA"
  npu.open().as_default()


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--data_path', type=str, default='./datasets', help='Directory path of dataset')
parser.add_argument('--train_epochs', default=400, type=int)
parser.add_argument('--model_dir', type=str, default='./checkpoint', help='save model')
############维测参数##############
parser.add_argument('--precision_mode', default="allow_mix_precision", type=str,help='the path to save over dump data')
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
parser.add_argument('--auto_tune', dest='auto_tune', type=ast.literal_eval, help='autotune flag, default is False')
args = parser.parse_args()


npu_config()
EPOCHS = args.train_epochs
BATCH_SIZE = args.batch_size

(train_images,train_labels),(_, _) = my_mnist.load_data(mnist_path=args.data_path,
                                                        get_new=False,
                                                        normalization=True,
                                                        one_hot=True,
                                                        detype=np.float32)
train_images = train_images.reshape(train_images.shape[0], 28, 28).astype('float32')

class Dense(layers.Layer):
    def __init__(self, input_dim, units):
        super(Dense,self).__init__()
        initializer = tf.initializers.glorot_uniform()
        # initializer = tf.initializers.glorot_normal()
        self.w = tf.Variable(initial_value=initializer(shape=(input_dim,units),dtype=tf.float32),trainable=True)
        self.b = tf.Variable(initial_value=tf.zeros(shape=(1,units),dtype=tf.float32),trainable=True)#节点的偏置也是行向量 才可以正常计算 即对堆叠的batch 都是加载单个batch内
    @tf.function
    def call(self,x,training=True):
        if training == True:
            y = tf.matmul(x,self.w)+self.b
            return y
        else:
            y = tf.matmul(x,self.w)+self.b
            return y
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dense1 = Dense(28*28+10,128)
        self.dense2 = Dense(128,1)
    @tf.function
    def call(self,x,y,training=True):
        """
        batch*dim+batch*10 在index_1维度组合 其余维度不变
        """
        x = tf.reshape(x,[-1,784]) #reshape 不改变原始的元素顺序 这很重要 防止变形时变成转置 忽略batch大小  只关注后面的维度一致
        x = tf.concat([x,y],axis=1)
        if training == True:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2_out = tf.nn.sigmoid(self.dense2(l1_out,training))
            return l2_out
        else:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2_out = tf.nn.sigmoid(self.dense2(l1_out,training))
            return l2_out

d = Discriminator()
x = train_images[0:2, :, :]
y = train_labels[0:2,:]

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator,self).__init__()
        self.dense1 = Dense(100+10,128)
        self.dense2 = Dense(128,784)
    @tf.function
    def call(self,x,y,training=True):
        x = tf.concat([x,y],axis=1)
        if training == True:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2 = tf.nn.sigmoid(self.dense2(l1_out,training))
        else:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2 = tf.nn.sigmoid(self.dense2(l1_out,training))
        l2_out = tf.reshape(l2,[-1,28,28])
        return l2_out

g = Generator()
z = tf.random.normal((1,100))
lable = tf.reshape(train_labels[0,:],(1,10))
image = g(z,lable,training=False)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
def d_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
def g_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

z_dim = 100
num_examples_to_generate = 100
seed = tf.random.uniform([num_examples_to_generate, z_dim],-1.0,1.0)
num_list = []
for i in range(10):
    num_list += [i]*10

seed_lable= tf.one_hot(num_list,depth=10,on_value=1.0,off_value=0.0,axis=-1,dtype=tf.float32) #axis理解成我们加入的深度10 在最终结果中的轴序号

seed = [seed,seed_lable]
@tf.function
def train_step(images,labels):
    # z = tf.random.normal([images.shape[0], z_dim],mean=0.0,stddev=1.0)
    z = tf.random.uniform([images.shape[0], z_dim],-1.0,1.0)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = g(z,labels,training=True)
        real_output = d(images,labels,training=True)
        fake_output = d(generated_images,labels,training=True)
        gen_loss = g_loss(fake_output)
        disc_loss = d_loss(real_output,fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, g.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, d.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, g.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, d.trainable_variables))
    return gen_loss, disc_loss

def train(train_images,train_labels,epochs):
    index = list(range(train_images.shape[0]))
    np.random.shuffle(index)
    train_images = train_images[index]
    train_labels = train_labels[index]
    images_batches = iter(tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE))
    labels_batches = iter(tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE))
    for epoch in range(epochs):
        start = time.time()
        while True:
            try:
                x_real_bacth = next(images_batches)
                y_label_bacth = next(labels_batches)
                start_time = time.time()
                gen_loss, disc_loss = train_step(x_real_bacth,y_label_bacth)
                total_time = time.time() - start_time
            except StopIteration:
                del images_batches
                del labels_batches
                np.random.shuffle(index)
                train_images = train_images[index]
                train_labels = train_labels[index]
                images_batches = iter(tf.data.Dataset.from_tensor_slices(train_images).batch(BATCH_SIZE))
                labels_batches = iter(tf.data.Dataset.from_tensor_slices(train_labels).batch(BATCH_SIZE))
                break

        print ('Time for epoch {} gen_loss: {:.4f} disc_loss: {:.4f} s/step: {:.4f}'.format(epoch + 1, gen_loss, disc_loss, total_time))

train(train_images,train_labels,EPOCHS)
g.save_weights(filepath=os.path.join(args.model_dir, 'tf_model'), save_format='tf')
