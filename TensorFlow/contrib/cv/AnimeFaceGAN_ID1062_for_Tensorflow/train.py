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
from ops import Hinge_loss, ortho_reg
import tensorflow as tf
import numpy as np
from utils import truncated_noise_sample, get_one_batch, session_config, read_images, check_dir
import cv2
import datetime
import scipy.io as sio
import argparse
import os
from generate_fake_img import generate_img, generate_img_by_class
from calc_IS_FID import get_IS, get_FID

parser = argparse.ArgumentParser()
# platform arguments (Huawei Ascend)
parser.add_argument("--chip", type=str, default="gpu", help="run on which chip, cpu or gpu or npu")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use (leave blank for CPU only)")
parser.add_argument("--platform", type=str, default="linux", help="Run on linux/apulis/modelarts platform. Modelarts "
                                                                   "Platform has some extra data copy operations")
parser.add_argument("--obs_dir", type=str, default="obs://lianlio/log", help="obs result path, not need on gpu and apulis platform")
parser.add_argument("--profiling", action="store_true", help="profiling for performance or not")
# data arguments
parser.add_argument("--dataset", type=str, default="../dataset", help="dataset path")
parser.add_argument("--output", type=str, default="../output", help="output path")
parser.add_argument("-c", "--num_classes", type=int, default=10, help="number of classes")
parser.add_argument("--img_h", type=int, default=32, help="image height")
parser.add_argument("--img_w", type=int, default=32, help="image width")
parser.add_argument("--train_img_size", type=int, default=32, help="image will be resized to this size when training")
parser.add_argument("--data", type=str, default="cifar10", help="which dataset to use (cifar10 / imagenet64)")
# metrics arguments
parser.add_argument("--metrics", type=str, default="fid", help="use FID or IS as metrics (fid / is)")
parser.add_argument("--precalculated_path", type=str, default="./metrics/res/stats_tf/fid_stats_cifar10_train.npz",
                    help="precalculated statistics for datasets, used in FID")
parser.add_argument("--gen_num", type=int, default=5000, help="number of generated images to calc IS or FID "
                                                              "(at least 2048 for FID)")
# training arguments
parser.add_argument('--use_fp16', action="store_true", help='enable mixed precision training')
parser.add_argument("--load_model", action="store_true", help="load model and continue to train")
parser.add_argument("--save_freq", type=int, default=1000, help="frequency of saving model")
parser.add_argument("--log_freq", type=int, default=50, help="frequency of logging")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size (larger batch size may have better performance)")
parser.add_argument("-i", "--train_itr", type=int, default=100000, help="number of training iterations")
parser.add_argument("--d_lr", type=float, default=4e-4, help="learning rate for discriminator")
parser.add_argument("--g_lr", type=float, default=1e-4, help="learning rate for generator")
parser.add_argument("--d_train_step", type=int, default=2, help="number of D training steps per G training step")
parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for Adam optimizer')
# model arguments
parser.add_argument("--base_channel", type=int, default=96, help="base channel number for G and D")
parser.add_argument("--z_dim", type=int, default=120, help="latent space dimensionality")
parser.add_argument("--shared_dim", type=int, default=128, help="shared embedding dimensionality")
parser.add_argument("--beta", type=float, default=1e-4, help="orthogonal regularization strength")
parser.add_argument("--truncation", type=float, default=2.0, help="truncation threshold")
parser.add_argument("--ema_decay", type=float, default=0.9999, help="decay rate of exponential moving average for the weights of G")
# other arguments
parser.add_argument("--debug", action="store_true", help="debug or not")
args = parser.parse_args()

if args.chip == "npu":
    from npu_bridge.npu_init import *
if args.debug is True:
    from tensorflow.python import debug as tf_dbg

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.metrics = args.metrics.upper()

# use different architectures for different image sizes
if args.train_img_size == 128:
    from networks_128 import Generator, Discriminator
elif args.train_img_size == 64:
    from networks_64 import Generator, Discriminator
elif args.train_img_size == 32:
    from networks_32 import Generator, Discriminator

# get current time
now = datetime.datetime.now()
now_str = now.strftime('%Y_%m_%d_%H_%M_%S')

# check output dir
model_path = os.path.join(args.output, "model", str(args.train_img_size))
resume_path = os.path.join(model_path, "model.ckpt")
ema_model_path = os.path.join(model_path, "ema.ckpt")
log_path = os.path.join(args.output, "log", str(args.train_img_size))
test_path = os.path.join(args.output, "gen_img")
fake_img_path = os.path.join(test_path, "fake", str(args.train_img_size))
image_of_each_class_path = os.path.join(test_path, "image_of_each_class", str(args.train_img_size))
check_dir(model_path)
check_dir(log_path)
if args.profiling is True:
    args.profiling_dir = "/tmp/profiling"
    check_dir(args.profiling_dir)


def train():
    train_phase = tf.Variable(tf.constant(True, dtype=tf.bool), name="train_phase")
    # train_phase = tf.placeholder(tf.bool)                           # is training or not
    x = tf.placeholder(tf.float32, [None, args.train_img_size, args.train_img_size, 3])             # input image(, which will be resized to 128x128)
    z = tf.placeholder(tf.float32, [None, args.z_dim])                  # latent vector
    y = tf.placeholder(tf.int32, [None])                                # class info

    with tf.variable_scope("generator"):
        embed_w = tf.get_variable("embed_w", [args.num_classes, args.shared_dim], initializer=tf.orthogonal_initializer()) # weight for shared embedding

    global_step = tf.Variable(0, trainable=False)                       # global training step
    add_step = global_step.assign(global_step + 1)

    set_train_phase_true = tf.assign(train_phase, True)
    set_train_phase_false = tf.assign(train_phase, False)

    G = Generator('generator', args.base_channel)
    D = Discriminator('discriminator', args.base_channel)
    fake_img = G(z, train_phase, y, embed_w, args.num_classes)                 # generate fake img
    fake_logits = D(fake_img, train_phase, y, args.num_classes, None)              # D(G(z), y)
    real_logits = D(x, train_phase, y, args.num_classes, 'NO_OPS')                 # D(x, y)

    D_loss, G_loss = Hinge_loss(real_logits, fake_logits)
    G_ortho = args.beta * ortho_reg(G.var_list())                       # Orthogonal Regularization
    G_loss += G_ortho                                                   # get total loss

    D_opt = tf.train.AdamOptimizer(args.d_lr, beta1=args.beta1, beta2=args.beta2).minimize(D_loss, var_list=D.var_list())
    G_opt = tf.train.AdamOptimizer(args.g_lr, beta1=args.beta1, beta2=args.beta2).minimize(G_loss, var_list=G.var_list())

    # loss scale for mixed precision training
    # if args.use_fp16 is True and args.chip == "npu":
    #     loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
    #                                                            decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    #     D_opt = NPULossScaleOptimizer(tf.train.AdamOptimizer(args.d_lr, beta1=args.beta1, beta2=args.beta2), loss_scale_manager).minimize(D_loss, var_list=D.var_list())
    #     G_opt = NPULossScaleOptimizer(tf.train.AdamOptimizer(args.g_lr, beta1=args.beta1, beta2=args.beta2), loss_scale_manager).minimize(G_loss, var_list=G.var_list())

    # add exponential moving average for G's weights
    with tf.variable_scope("ema_weights"):
        var_ema = tf.train.ExponentialMovingAverage(args.ema_decay, global_step)
        with tf.control_dependencies([G_opt]):
            G_opt_ema = var_ema.apply(tf.trainable_variables(scope='generator'))
        # assign ema weights
        assign_vars = []
        for var in tf.trainable_variables(scope='generator'):
            v = var_ema.average(var)
            if v is not None:
                assign_vars.append(tf.assign(var, v))

    with tf.variable_scope("metrics", reuse=tf.AUTO_REUSE):
        FID_now = tf.get_variable("FID_now", shape=[], initializer=tf.constant_initializer(1e3), trainable=False)
        IS_now = tf.get_variable("IS_now", shape=[], initializer=tf.constant_initializer(0.0), trainable=False)
        FID_best = tf.get_variable("FID_best", shape=[], initializer=tf.constant_initializer(1e3), trainable=False)
        IS_best = tf.get_variable("IS_best", shape=[], initializer=tf.constant_initializer(0.0), trainable=False)

    # log loss, FID, IS
    log_suffix = "_" + str(args.train_img_size) + "_bs_" + str(args.batch_size) + "_ch_" + str(args.base_channel)
    tf.summary.scalar(now_str + '/d_loss' + log_suffix, D_loss)
    tf.summary.scalar(now_str + '/g_loss' + log_suffix, G_loss)
    # tf.summary.scalar(now_str + '/IS' + log_suffix, IS_now)
    # tf.summary.scalar(now_str + '/FID' + log_suffix, FID_now)
    summary_op = tf.summary.merge_all()

    config = session_config(args)

    print("Using", args.chip, "!")

    if args.data == "cifar10":
        # get cifar-10 training data
        data_path = os.path.join(args.dataset, "data_batch_")
        test_data_path = os.path.join(args.dataset, "test_batch.mat")
        raw_data = np.concatenate((sio.loadmat(data_path + "1.mat")["data"],
                               sio.loadmat(data_path + "2.mat")["data"],
                               sio.loadmat(data_path + "3.mat")["data"],
                               sio.loadmat(data_path + "4.mat")["data"],
                               sio.loadmat(data_path + "5.mat")["data"],
                               sio.loadmat(test_data_path)["data"]
                               ),
                              axis=0)
        raw_data = np.reshape(raw_data, [-1, 3, args.img_h, args.img_w])
        raw_data = np.transpose(raw_data, axes=[0, 2, 3, 1])  # (N, H, W, C)
        labels = np.concatenate((sio.loadmat(data_path + "1.mat")["labels"],
                                 sio.loadmat(data_path + "2.mat")["labels"],
                                 sio.loadmat(data_path + "3.mat")["labels"],
                                 sio.loadmat(data_path + "4.mat")["labels"],
                                 sio.loadmat(data_path + "5.mat")["labels"],
                                 sio.loadmat(test_data_path)["labels"]
                                 ),
                                axis=0)[:, 0]
    elif args.data == "imagenet64":
        # get imagenet64 training data
        data_path = os.path.join(args.dataset, "imagenet64.mat")
        data_and_label = sio.loadmat(data_path)
        labels = data_and_label["labels"][0, :]
        raw_data = data_and_label["data"]
    else:
        pass

    # resize images to training size
    start = datetime.datetime.now()
    data = np.zeros(shape=[raw_data.shape[0], args.train_img_size, args.train_img_size, 3], dtype=raw_data.dtype)
    for i, img in enumerate(raw_data):
        data[i] = cv2.resize(img, dsize=(args.train_img_size, args.train_img_size), interpolation=cv2.INTER_LINEAR)
    end = datetime.datetime.now()
    print("data preprocess time:", (end - start).total_seconds())

    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter(logdir=log_path, graph=sess.graph)
        sess.run(tf.global_variables_initializer())

        if args.debug is True:
            sess = tf_dbg.LocalCLIDebugWrapperSession(sess)

        # load model
        saver = tf.train.Saver()
        if args.load_model is True:
            print('Loading checkpoint from {}...'.format(resume_path))
            saver.restore(sess, save_path=resume_path)

        for itr in range(args.train_itr):
            d_update_time = 0       # discriminator update time
            g_update_time = 0       # generator update time
            data_preprocess_time = 0

            # Train Discriminator
            for d in range(args.d_train_step):
                # read one mini-batch
                start = datetime.datetime.now()
                batch, Y = get_one_batch(data, labels, args.batch_size)    # get one batch
                end = datetime.datetime.now()
                data_preprocess_time += (end - start).total_seconds()

                # truncation trick
                Z = truncated_noise_sample(args.batch_size, args.z_dim, args.truncation)

                start = datetime.datetime.now()
                sess.run(set_train_phase_true)
                sess.run(D_opt, feed_dict={z: Z, x: batch, y: Y})
                end = datetime.datetime.now()
                d_update_time += (end - start).total_seconds()

            # Train Generator
            Z = truncated_noise_sample(args.batch_size, args.z_dim, args.truncation)
            start = datetime.datetime.now()
            sess.run(set_train_phase_true)
            sess.run([G_opt_ema, add_step, global_step], feed_dict={z: Z, y: Y})
            end = datetime.datetime.now()
            g_update_time += (end - start).total_seconds()

            if itr % args.log_freq == 0:
                sess.run(set_train_phase_false)
                summary, d_loss, g_loss, is_now, is_best, fid_now, fid_best = sess.run([summary_op, D_loss, G_loss, IS_now, IS_best, FID_now, FID_best],
                                                   feed_dict={z: Z, x: batch, y: Y})
                summary_writer.add_summary(summary, itr)
                metrics_best = fid_best if args.metrics == "FID" else is_best
                # print("Iteration: %d, D_loss: %f, G_loss: %f, IS: %f, FID: %f, best %s: %f, "
                #       "D_updata_time: %f(s), G_updata_time: %f(s), data preprocess time: %f(s)"
                #       % (itr, d_loss, g_loss, is_now, fid_now, args.metrics, metrics_best,
                #          d_update_time, g_update_time, data_preprocess_time))
                print("Iteration: %d, D_loss: %f, G_loss: %f, "
                      "D_updata_time: %f(s), G_updata_time: %f(s), data preprocess time: %f(s)"
                      % (itr, d_loss, g_loss, d_update_time, g_update_time, data_preprocess_time))
                # generate fake images for each class
                generate_img_by_class(args, image_of_each_class_path, sess, fake_img, z, y)

                # print loss scale value
                if args.use_fp16 is True and args.chip == "npu":
                    lossScale = tf.get_default_graph().get_tensor_by_name("loss_scale:0")
                    overflow_status_reduce_all = tf.get_default_graph().get_tensor_by_name(
                        "overflow_status_reduce_all:0")
                    l_s, overflow_status_reduce_all = sess.run([lossScale, overflow_status_reduce_all])
                    print('loss_scale is: ', l_s)
                    print("overflow_status_reduce_all:", overflow_status_reduce_all)
            if itr % args.save_freq == 0:
                saver.save(sess, save_path=resume_path)         # save current model
                print("Model saved in", resume_path)
                sess.run(set_train_phase_false)
                sess.run(assign_vars, feed_dict={z: Z, y: Y})   # get ema model

                # calc FID and IS
                # generate_img(args, fake_img_path, sess, fake_img, z, y)  # generate fake images
                # images_list = read_images(fake_img_path)
                # images = np.array(images_list).astype(np.float32)

                # fid_now = get_FID(images, args)
                # is_now, _ = get_IS(images_list, args, splits=10)
                #
                # if args.metrics == "FID":
                #     fid_best = sess.run(FID_best)
                #     if fid_now < fid_best:
                #         fid_best = fid_now
                #         saver.save(sess, save_path=ema_model_path)  # save ema model
                #         print("New best model!\nBest FID:", fid_best)
                # else:
                #     is_best = sess.run(IS_best)
                #     if is_now > is_best:
                #         is_best = is_now
                #         saver.save(sess, save_path=ema_model_path)  # save ema model
                #         print("New best model!\nBest IS:", is_best)
                saver.save(sess, save_path=ema_model_path)   # save ema model
                print("EMA Model saved in", ema_model_path)
                saver.restore(sess, save_path=resume_path)   # restore current model

                # if args.metrics == "FID":
                #     sess.run(tf.assign(FID_best, tf.cast(tf.constant(fid_best), tf.float32)))    # update best FID / IS
                # else:
                #     sess.run(tf.assign(IS_best, tf.cast(tf.constant(is_best), tf.float32)))
                #
                # sess.run(tf.assign(IS_now, tf.cast(tf.constant(is_now), tf.float32)))          # update FID and IS
                # sess.run(tf.assign(FID_now, tf.cast(tf.constant(fid_now), tf.float32)))

    summary_writer.close()

    if args.platform.lower() == 'modelarts':
        from help_modelarts import modelarts_result2obs
        modelarts_result2obs(args)
        print("Data transferred to OBS!")

    print("Training finished!")


if __name__ == "__main__":
    train()
