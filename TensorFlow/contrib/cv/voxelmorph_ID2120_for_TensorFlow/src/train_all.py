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

"""
train atlas-based alignment with CVPR2018 version of VoxelMorph
"""

# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser
import time

# third-party imports
import tensorflow as tf
import numpy as np
import tensorflow.python.keras as keras
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate
import nibabel as nib
from tensorflow.keras.callbacks import LearningRateScheduler
import math

# project imports
from datagenerators import zyh_data_in
import datagenerators
import networks
import losses

sys.path.append('../ext/neuron')
import neuron.callbacks as nrn_gen

from npu_bridge.npu_init import *


def train(train_data_dir,
          atlas_file,
          model_dir,
          lr,
          nb_epochs,
          reg_param,
          batch_size,
          load_model_file,
          tensorboard_log_dir,
          ):
    """
    model training function
    :param data_dir: folder with npz files for each subject.
    :param atlas_file: atlas filename. So far we support npz file with a 'vol' variable
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param model_dir: the model directory to save to
    :param gpu_id: integer specifying the gpu to use
    :param lr: learning rate
    :param n_iterations: number of training iterations
    :param reg_param: the smoothness/reconstruction tradeoff parameter (lambda in CVPR paper)
    :param steps_per_epoch: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    :param load_model_file: optional h5 model file to initialize with
    :param data_loss: data_loss: 'mse' or 'ncc
    """
    # load atlas from provided files. The atlas we used is 160x192x224.
    atlas_vol = nib.load(atlas_file).dataobj[np.newaxis, ..., np.newaxis]
    vol_size = atlas_vol.shape[1:-1]

    # prepare data files
    train_vol_names = glob.glob(os.path.join(train_data_dir, '*.nii.gz'))
    # random.shuffle(train_vol_names)  # shuffle volume list
    assert len(train_vol_names) > 0, "Could not find any training data"

    # UNET filters for voxelmorph-1 and voxelmorph-2,
    # these are architectures presented in CVPR 2018
    nf_enc = [16, 32, 32, 32]
    nf_dec = [32, 32, 32, 32, 32, 16, 16]

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # prepare log folder
    if not os.path.isdir(tensorboard_log_dir):
        os.mkdir(tensorboard_log_dir)
    # if not os.path.isdir(os.path.join(tensorboard_log_dir, 'plugins/profile')):
    #     os.mkdir(os.path.join(tensorboard_log_dir, 'plugins/profile'))

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("**********")

    # prepare the model
    src = tf.placeholder(dtype=tf.float32, shape=[batch_size, 160, 192, 224, 1])
    tgt = tf.placeholder(dtype=tf.float32, shape=[batch_size, 160, 192, 224, 1])
    y, flow = networks.cvpr2018_net(vol_size, nf_enc, nf_dec, src, tgt)  # vol_size, enc_nf, dec_nf, src, tgt
    residual = y - tgt
    loss_mse = tf.reduce_mean(residual**2)
    loss_flow = losses.Grad('l2').loss(y, flow)
    loss = loss_mse + reg_param * loss_flow
    global_step = tf.Variable(0, trainable=False)

    # optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    # opt loss scale
    opt = tf.train.AdamOptimizer(lr)
    # loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 64, incr_every_n_steps=1000,
    #                                                        decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    loss_scale_manager = FixedLossScaleManager(loss_scale=2 ** 32)
    opt = NPULossScaleOptimizer(opt, loss_scale_manager)
    optimizer = opt.minimize(loss, global_step=global_step)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('mse', loss_mse)
    tf.summary.scalar('loss_flow', loss_flow)
    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

    custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes("./src/fusion_switch.cfg")
    # config = npu_tf_config.session_dump_config(config, action='fusion_switch')
    # config = npu_tf_config.session_dump_config(config, action='overflow')

    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)

        train_writer = tf.summary.FileWriter(logdir=tensorboard_log_dir, graph=sess.graph)
        # saver is used to save the model
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        # saver.save(sess=sess, save_path=os.path.join(model_dir, "weight_init"))

        if load_model_file is not None:
            print('loading', load_model_file)
            saver.restore(sess, load_model_file)

        for epoch in range(nb_epochs):
            tf.logging.info("*********　　epoch %d　　　***********", epoch)
            train_loss_list = []
            train_loss_mse_list = []
            train_loss_flow_list = []
            for step, volname in enumerate(train_vol_names):
                lossScale = tf.get_default_graph().get_tensor_by_name("loss_scale:0")
                overflow_status_reduce_all = tf.get_default_graph().get_tensor_by_name("overflow_status_reduce_all:0")

                moving = zyh_data_in(volname)
                feed_dict = {src: moving, tgt: atlas_vol}

                # _, train_loss, train_loss_mse, train_loss_flow, summary = sess.run([optimizer, loss, loss_mse, loss_flow, summary_op], feed_dict=feed_dict)
                # train_loss_list.append(train_loss)

                t_start = time.time()

                _, train_loss, train_loss_mse, train_loss_flow, summary, l_s, overflow_status_reduce_all, global_steppp = sess.run(
                    [optimizer,
                     loss,
                     loss_mse,
                     loss_flow,
                     summary_op,
                     lossScale,
                     overflow_status_reduce_all,
                     global_step
                     ], feed_dict=feed_dict)
                     
                t_end = time.time()

                train_loss_list.append(train_loss)
                train_loss_mse_list.append(train_loss_mse)
                train_loss_flow_list.append(train_loss_flow)
                tf.logging.info("step {%d} --->  loss: {%.5f}, loss_mse: {%.5f}, loss_flow: {%.3e}, time: {%.3f}\n", step, train_loss, train_loss_mse, train_loss_flow, t_end-t_start)

                # print('loss_scale is: ', l_s)
                # print("overflow_status_reduce_all:", overflow_status_reduce_all)
                # print("global_step:", global_steppp)

            train_writer.add_summary(summary, epoch)
            tf.logging.info('**************************')
            tf.logging.info("train_loss = %s", np.mean(train_loss_list))
            tf.logging.info("train_loss_mse = %s", np.mean(train_loss_mse_list))
            tf.logging.info("train_loss_flow = %s", np.mean(train_loss_flow_list))
            tf.logging.info('**************************\n')

            saver.save(sess=sess, save_path=os.path.join(model_dir, "model"))


        train_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("train_data_dir", type=str,
                        help="train data folder")

    parser.add_argument("--atlas_file", type=str,
                        dest="atlas_file", default='../../Dataset-ABIDE/atlas_abide_brain_crop.nii.gz',
                        help="gpu id number")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='../models/',
                        help="models folder")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=50,
                        help="number of iterations")
    parser.add_argument("--lambda", type=float,
                        dest="reg_param", default=0.01,
                        help="regularization parameter")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=1,
                        help="batch_size")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default=None,
                        help="optional ckpt file to initialize with")
    parser.add_argument("--tensorboard_log_dir", type=str,
                        dest="tensorboard_log_dir", default='../log/')

    args = parser.parse_args()
    train(**vars(args))
