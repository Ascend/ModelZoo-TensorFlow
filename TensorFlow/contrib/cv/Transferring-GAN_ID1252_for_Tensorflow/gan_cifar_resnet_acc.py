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
"""WGAN-GP ResNet for CIFAR-10"""

import os, sys

sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.cifar10
import tflib.plot

import numpy as np
import tensorflow as tf
# import sklearn.datasets
import argparse
import time
import functools
import locale
import os
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

# for npu loss-scale 
from npu_bridge.npu_init import *


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default='', help='Pretrained model location')
    parser.add_argument('--data_path', type=str, default='./dataset', help='Datasets location')
    parser.add_argument('--output_path', type=str, default='./output', help='Output location,saving trained models')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument("--GEN_BS_MULTIPLE", type=int, default=2, help=" Generator batch size, as a multiple of BATCH_SIZE")
    parser.add_argument("--ITERS", type=int, default=1000, help=" How many iterations to train for")
    parser.add_argument("--DIM_G", type=int, default=128, help=" Generator dimensionality")
    parser.add_argument("--DIM_D", type=int, default=128, help=" Critic dimensionality")
    parser.add_argument("--NORMALIZATION_G", type=bool, default=True, help=" Use batchnorm in generator?")
    parser.add_argument("--NORMALIZATION_D", type=bool, default=False, help=" Use batchnorm (or layernorm) in critic?")
    parser.add_argument("--OUTPUT_DIM", type=int, default=3072, help=" Number of pixels in CIFAR10 (32*32*3)")
    parser.add_argument("--LR", type=int, default=2e-4, help=" Initial learning rate")
    parser.add_argument("--DECAY", type=bool, default=True, help=" Whether to decay LR over learning")
    parser.add_argument("--N_CRITIC", type=int, default=5, help=" Critic steps per generator steps")
    parser.add_argument("--INCEPTION_FREQUENCY", type=int, default=1, help=" How frequently to calculate Inception score")
    parser.add_argument("--CONDITIONAL", type=bool, default=True, help=" Whether to train a conditional or unconditional model")
    parser.add_argument("--ACGAN", type=bool, default=True, help=" If CONDITIONAL, whether to use ACGAN or vanilla conditioning")
    parser.add_argument("--ACGAN_SCALE", type=int, default=1., help=" How to scale the critic's ACGAN loss relative to WGAN loss")
    parser.add_argument("--ACGAN_SCALE_G", type=int, default=0.1, help=" How to scale generator's ACGAN loss relative to WGAN loss")
    return parser.parse_args()

args = parse_args()

DATA_CACHE_PATH = args.data_path
Inception_path = args.data_path+"/inception"
MODEL_CACHE_PATH = args.data_path+'/ckpt'

# print("\n"*2,os.listdir(inception_path),"\n"*2)
import tflib.inception_score


softmax = tflib.inception_score.init_inception(Inception_path)

# locale.setlocale(locale.LC_ALL, '')

DATA_DIR = DATA_CACHE_PATH
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')


BATCH_SIZE = args.batch_size  # Critic batch size
GEN_BS_MULTIPLE = args.GEN_BS_MULTIPLE
ITERS = args.ITERS
DIM_G = args.DIM_G
DIM_D = args.DIM_D
NORMALIZATION_G = args.NORMALIZATION_G
NORMALIZATION_D = args.NORMALIZATION_D
OUTPUT_DIM = args.OUTPUT_DIM
LR = args.LR
DECAY = args.DECAY
N_CRITIC = args.N_CRITIC
INCEPTION_FREQUENCY = args.INCEPTION_FREQUENCY
CONDITIONAL = args.CONDITIONAL
ACGAN = args.ACGAN
ACGAN_SCALE = args.ACGAN_SCALE
ACGAN_SCALE_G = args.ACGAN_SCALE_G

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print("WARNING! Conditional model without normalization in D might be effectively unconditional!")


def nonlinearity(x):
    return tf.nn.relu(x)


def Normalize(name, inputs, labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs, labels=labels, n_labels=10)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name, [0, 2, 3], inputs, labels=labels, n_labels=10)
        else:
            return lib.ops.batchnorm.Batchnorm(name, [0, 2, 3], inputs, fused=True)
    else:
        return inputs


def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output


def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample == 'up':
        conv_1 = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name + '.N1', output, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name + '.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name + '.N2', output, labels=labels)
    output = nonlinearity(output)
    output = conv_2(name + '.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output


def OptimizedResBlockDisc1(inputs):
    conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
    conv_2 = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False,
                             biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)
    output = nonlinearity(output)
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output


def Generator(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([int(n_samples), 128])
    output = lib.ops.linear.Linear('Generator.Input', 128, 4 * 4 * DIM_G, noise)
    output = tf.reshape(output, [-1, DIM_G, 4, 4])
    output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])


def Discriminator(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2, 3])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if CONDITIONAL and ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, 10, output)
        return output_wgan, output_acgan
    else:
        return output_wgan, None

    
_iteration = tf.placeholder(tf.int32, shape=None)
all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

labels_splits = all_real_labels

fake_data_splits = Generator(BATCH_SIZE, labels_splits)

all_real_data = tf.reshape(2 * ((tf.cast(all_real_data_int, tf.float32) / 256.) - .5), [BATCH_SIZE, OUTPUT_DIM])
all_real_data += tf.random_uniform(shape=[BATCH_SIZE, OUTPUT_DIM], minval=0., maxval=1. / 128)  # dequantize
all_real_data_splits = all_real_data

disc_costs = []
disc_acgan_costs = []
disc_acgan_accs = []
disc_acgan_fake_accs = []

real_and_fake_data = tf.concat([
    all_real_data_splits,
    fake_data_splits,
], axis=0)
real_and_fake_labels = tf.concat([
    labels_splits,
    labels_splits
], axis=0)

disc_all, disc_all_acgan = Discriminator(real_and_fake_data, real_and_fake_labels)

disc_real = disc_all[:BATCH_SIZE]
disc_fake = disc_all[BATCH_SIZE:]
disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))
if CONDITIONAL and ACGAN:
    disc_acgan_costs.append(tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_all_acgan[:BATCH_SIZE],
                                                        labels=real_and_fake_labels[:BATCH_SIZE])
    ))
    disc_acgan_accs.append(tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.to_int32(tf.argmax(disc_all_acgan[:BATCH_SIZE], dimension=1)),
                real_and_fake_labels[:BATCH_SIZE]
            ),
            tf.float32
        )
    ))
    disc_acgan_fake_accs.append(tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.to_int32(tf.argmax(disc_all_acgan[BATCH_SIZE:], dimension=1)),
                real_and_fake_labels[BATCH_SIZE:]
            ),
            tf.float32
        )
    ))

real_data = tf.concat([all_real_data_splits, all_real_data_splits], axis=0)
fake_data = tf.concat([fake_data_splits, fake_data_splits], axis=0)
labels = tf.concat([
    labels_splits,
], axis=0)
alpha = tf.random_uniform(
    shape=[BATCH_SIZE * 2, 1],
    minval=0.,
    maxval=1.
)
differences = fake_data - real_data
interpolates = real_data + (alpha * differences)
gradients = tf.gradients(Discriminator(interpolates, labels)[0], [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = 10 * tf.reduce_mean((slopes - 1.) ** 2)
disc_costs.append(gradient_penalty)

disc_wgan = tf.add_n(disc_costs)
if CONDITIONAL and ACGAN:
    disc_acgan = tf.add_n(disc_acgan_costs)
    disc_acgan_acc = tf.add_n(disc_acgan_accs)
    disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs)
    disc_cost = disc_wgan + (ACGAN_SCALE * disc_acgan)
else:
    disc_acgan = tf.constant(0.)
    disc_acgan_acc = tf.constant(0.)
    disc_acgan_fake_acc = tf.constant(0.)
    disc_cost = disc_wgan

disc_params = lib.params_with_name('Discriminator.')

if DECAY:
    decay = tf.maximum(0., 1. - (tf.cast(_iteration, tf.float32) / ITERS))
else:
    decay = 1.

gen_costs = []
gen_acgan_costs = []
n_samples = GEN_BS_MULTIPLE * BATCH_SIZE
fake_labels = tf.cast(tf.random_uniform([n_samples]) * 10, tf.int32)
if CONDITIONAL and ACGAN:
    disc_fake, disc_fake_acgan = Discriminator(Generator(n_samples, fake_labels), fake_labels)
    gen_costs.append(-tf.reduce_mean(disc_fake))
    gen_acgan_costs.append(tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
    ))
else:
    gen_costs.append(-tf.reduce_mean(Discriminator(Generator(n_samples, fake_labels), fake_labels)[0]))
gen_cost = (tf.add_n(gen_costs))
if CONDITIONAL and ACGAN:
    gen_cost += (ACGAN_SCALE_G * (tf.add_n(gen_acgan_costs)))

rate = 2**23
gen_opt = tf.train.AdamOptimizer(learning_rate=LR * decay, beta1=0., beta2=0.9)
disc_opt = tf.train.AdamOptimizer(learning_rate=LR * decay, beta1=0., beta2=0.9)
gen_gv = gen_opt.compute_gradients(gen_cost , var_list=lib.params_with_name('Generator'))
disc_gv = disc_opt.compute_gradients(disc_cost , var_list=disc_params)
gen_gv_ls = [(g,v / rate) for g, v in gen_gv ]
disc_gv_ls= [(g,v / rate) for g, v in disc_gv ]
gen_train_op = gen_opt.apply_gradients(gen_gv)
disc_train_op = disc_opt.apply_gradients(disc_gv)

# gen_loss_scale_manager = FixedLossScaleManager(loss_scale=rate)
# disc_loss_scale_manager= FixedLossScaleManager(loss_scale=rate)
# gen_opt = NPULossScaleOptimizer(gen_opt,gen_loss_scale_manager)
# disc_opt= NPULossScaleOptimizer(disc_opt,disc_loss_scale_manager)
# # 为了使用loss scale 改写 loss optimizer  => 没法加载预训练 
# gen_train_op = gen_opt.minimize(gen_cost,var_list=lib.params_with_name('Generator'))
# disc_train_op= disc_opt.minimize(disc_cost, var_list=disc_params)

# Function for generating samples
frame_i = [0]
fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
fixed_labels = tf.constant(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32'))
fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise)


def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    samples = ((samples + 1.) * (255. / 2)).astype('int32')
    lib.save_images.save_images(samples.reshape((100, 3, 32, 32)), 'samples_{}.png'.format(frame))


# Function for calculating inception score
fake_labels_100 = tf.cast(tf.random_uniform([100]) * 10, tf.int32)
samples_100 = Generator(100, fake_labels_100)


def get_inception_score(n):
    print("get_inception_score")
    all_samples = []
    for i in range(n // 100):
        all_samples.append(session.run(samples_100))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples + 1.) * (255.99 / 2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return lib.inception_score.get_inception_score(list(all_samples), softmax)


train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, DATA_DIR)


def inf_train_gen():
    while True:
        for images, _labels in train_gen():
                yield images, _labels


# some config on this
# config = tf.ConfigProto(allow_soft_placement=True)
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训练
custom_op.parameter_map["precision_mode"].s=tf.compat.as_bytes("allow_mix_precision")

# config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关
# fp_point: Sub
# gradients_2/AddN_49
# custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/HwHiAiUser/zhegongda/transfergan/profiling","training_trace":"on","l2":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":"gradients_2/AddN_49","aic_metrics":"PipeUtilization"}') 
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF #关闭remap开关

config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:
    
    # tf.io.write_graph(session.graph,"./graph","graph.pbtxt")
    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()
    saver = tf.train.Saver(max_to_keep=20)
    start_time = time.time()
    saver.restore(session,"{}/transfer-gan-inception-score-8.068276405334473-inception-std-0.08256248384714127-15999".format(MODEL_CACHE_PATH))
    print("load pretrained model  time cost: {}".format(time.time()- start_time))
    for iteration in range(ITERS):
        start_time = time.time()
        print("start")
        # if iteration > 0:
        #     _ = session.run([gen_train_op], feed_dict={_iteration: iteration})

        for i in range(N_CRITIC):
            step_start_time = time.time()
            _data, _labels = next(gen)
            if CONDITIONAL and ACGAN:
                _disc_cost, _disc_wgan, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _ = session.run(
                    [disc_cost, disc_wgan, disc_acgan, disc_acgan_acc, disc_acgan_fake_acc, disc_train_op],
                    feed_dict={all_real_data_int: _data, all_real_labels: _labels, _iteration: iteration})
                print("time: {}".format(time.time()-step_start_time))
            else:
                _disc_cost, _ = session.run([disc_cost, disc_train_op],
                                            feed_dict={all_real_data_int: _data, all_real_labels: _labels,
                                                       _iteration: iteration})
                print("time2: {}".format(time.time()-step_start_time))
        print("step iteration {} sec/step : {}".format(iteration,time.time()-start_time))
        # save

        # break
        print("wgan :{} , acgan: {} acc_real: {} acc_fake:{}".format(_disc_wgan, _disc_acgan,_disc_acgan_acc,_disc_acgan_fake_acc))

        if iteration % INCEPTION_FREQUENCY == INCEPTION_FREQUENCY - 1:
            print("start evaluation ")
            inception_score = get_inception_score(50000)
            # saver.save(session, "./model/transfer-gan-inception-score-{}-inception-std-{}".format(inception_score[0],
            #                                                                                       inception_score[1]),
            #            global_step=iteration)
            print("Final Average Distances : inception_50k score : {} inception_50k_std: {}".format(inception_score[0],inception_score[1]))
        break