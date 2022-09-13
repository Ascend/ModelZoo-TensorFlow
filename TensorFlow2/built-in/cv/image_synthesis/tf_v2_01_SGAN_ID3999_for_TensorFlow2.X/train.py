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
SGAN  标准GAN  standard GAN 
损失函数: 经典log损失 即交叉熵损失 对抗损失 赋予正负样本 1 0 标签后的推导值 其实就是交叉熵 只是分开了正负样本
        但是基于tensorflow的交叉熵计算时优化过的 对于log(0)不会出现无穷值 这就是我自己的log函数容易崩坏的原因
网络结构: MLP 至少有两层 即输入层后 至少1个中间层 然后是输出层 常用128节点 
数据形式: 不带卷积 没有深度维  图片压缩到0 1 之间 
生成器: sigmoid 映射到0 1 之间 迎合数据格式
判别器: sigmoid 映射到0 1 之间 迎合loss公式的约束
初始化: xavier初始化  即考虑输入输出维度的 glorot uniform
训练： 判别器和生成器同时训练 同步训练 不偏重任一方
"""
import time
import ast
import argparse
import my_mnist  
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import npu_device

def npu_config():
  if args.data_dump_flag:
    npu_device.global_options().dump_config.enable_dump = True
    npu_device.global_options().dump_config.dump_path = args.data_dump_path
    npu_device.global_options().dump_config.dump_step = args.data_dump_step
    npu_device.global_options().dump_config.dump_mode = "all"

  if args.over_dump:
    npu_device.global_options().dump_config.enable_dump_debug = True
    npu_device.global_options().dump_config.dump_path = args.over_dump_path
    npu_device.global_options().dump_config.dump_debug_mode = "all"

  if args.profiling:
    npu_device.global_options().profiling_config.enable_profiling = True
    profiling_options = '{"output":"' + args.profiling_dump_path + '", \
                        "training_trace":"on", \
                        "task_trace":"on", \
                        "aicpu":"on", \
                        "L2": "on", \
                        "aic_metrics":"PipeUtilization",\
                        "fp_point":"", \
                        "bp_point":""}'
    npu_device.global_options().profiling_config.profiling_options = profiling_options
  npu_device.global_options().precision_mode=args.precision_mode
  if args.use_mixlist and args.precision_mode=='allow_mix_precision':
    npu_device.global_options().modify_mixlist=args.mixlist_file
  if args.fusion_off_flag:
    npu_device.global_options().fusion_switch_file=args.fusion_off_file
  if args.auto_tune:
    npu_device.global_options().auto_tune_mode="RL,GA"
  npu_device.open().as_default()



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--data_path', type=str, default='./', help='Directory path of S3DIS dataset')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--lossscale', default=2, help='loss scale for mix precision')
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
############多p参数##############
parser.add_argument("--rank_size", default=1, type=int, help="rank size")
parser.add_argument("--device_id", default=0, type=int, help="Ascend device id")
args = parser.parse_args()

npu_config()
epochs = args.epochs
data_path = args.data_path
batch_size = args.batch_size

(train_images,train_labels),(_, _) = my_mnist.load_data(data_path=data_path, get_new=False,
                                                        normalization=True,
                                                        one_hot=True,
                                                        detype=np.float32)
if args.rank_size !=1:
    train_images = np.split(train_images, args.rank_size)
    train_images = train_images[args.device_id]

    train_labels = np.split(train_labels, args.rank_size)
    train_labels = train_labels[args.device_id]

    batch_size = batch_size // args.rank_size

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

# x = tf.random.normal(shape=(64,784))
# a = Dense(28*28,128)

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.dense1 = Dense(28*28,128)
        self.dense2 = Dense(128,1)
    @tf.function
    def call(self,x,training=True):
        x = tf.reshape(x,[-1,784]) #reshape 不改变原始的元素顺序 这很重要 防止变形时变成转置 忽略batch大小  只关注后面的维度一致
        if training == True:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2_out = tf.nn.sigmoid(self.dense2(l1_out,training))
            return l2_out
        else:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2_out = tf.nn.sigmoid(self.dense2(l1_out,training))
            return l2_out


d = Discriminator()
# x = train_images[0:2, :, :]

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator,self).__init__()
        self.dense1 = Dense(100,128)
        self.dense2 = Dense(128,784)
    @tf.function
    def call(self,x,training=True):
        if training == True:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2 = tf.nn.sigmoid(self.dense2(l1_out,training))
        else:
            l1_out = tf.nn.relu(self.dense1(x,training))
            l2 = tf.nn.sigmoid(self.dense2(l1_out,training))
        l2_out = tf.reshape(l2,[-1,28,28])
        return l2_out

g = Generator()
# z = tf.random.normal((1,100))

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def d_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def g_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

if args.rank_size != 1:
    generator_optimizer = npu_device.distribute.npu_distributed_keras_optimizer_wrapper(tf.keras.optimizers.Adam(1e-4))
    discriminator_optimizer = npu_device.distribute.npu_distributed_keras_optimizer_wrapper(tf.keras.optimizers.Adam(1e-5))
else:
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)


if args.rank_size !=1:
    training_vars = g.trainable_variables
    npu_device.distribute.broadcast(training_vars, root_rank=0)

    training_vars2 = d.trainable_variables
    npu_device.distribute.broadcast(training_vars2, root_rank=0)

# print("use fixed loss scale")
# generator_optimizer = npu_device.train.optimizer.NpuLossScaleOptimizer(generator_optimizer, dynamic=False, initial_scale=32768.)
# discriminator_optimizer = npu_device.train.optimizer.NpuLossScaleOptimizer(discriminator_optimizer, dynamic=False, initial_scale=32768.)

z_dim = 100
# num_examples_to_generate = 100
# seed = tf.random.uniform([num_examples_to_generate, z_dim],-1.0,1.0)

z = tf.random.uniform([batch_size, z_dim],-1.0,1.0)
@tf.function
def train_step(images,labels):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = g(z,training=True)
        real_output = d(images,training=True)
        fake_output = d(generated_images,training=True)
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
    images_batches = iter(tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size))
    labels_batches = iter(tf.data.Dataset.from_tensor_slices(train_labels).batch(batch_size))
    for epoch in range(epochs):
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
                images_batches = iter(tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size))
                labels_batches = iter(tf.data.Dataset.from_tensor_slices(train_labels).batch(batch_size))
                break
        print ('Time for epoch {} gen_loss: {:.4f} disc_loss: {:.4f} s/step: {:.4f}'.format(epoch + 1, gen_loss, disc_loss, total_time))

train(train_images,train_labels,epochs)