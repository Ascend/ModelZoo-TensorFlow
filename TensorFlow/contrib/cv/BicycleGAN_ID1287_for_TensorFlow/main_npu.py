
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

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys
import argparse
import numpy as np
import tensorflow as tf
from model_npu_tmp import BicycleGAN
from folder_npu import check_folder
# from load_data import load_images
import os
# import moxing as mox
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = "3"


def parse_args():
    desc = "Tensorflow implementation of BicycleGAN"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--Z_dim', type=int, default=8, help='Size of latent vector')
    parser.add_argument('--reconst_coeff', type=float, default=10, help='Reconstruction Coefficient')
    parser.add_argument('--latent_coeff', type=float, default=0.5, help='Latent Coefficient')
    parser.add_argument('--kl_coeff', type=float, default=0.01, help='KL Coefficient')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning Rate')
    parser.add_argument('--image_size', type=int, default=256, help='Image Size')
    parser.add_argument('--batch_size', type=int, default=1, help='Size of the minibatch')
    parser.add_argument('--gan_type', type=str, default='BicycleGAN', help='Type of GAN')
    parser.add_argument('--dataset', type=str, default='./maps', help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=200, help='The number of epochs to run')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--train_url', type=str, default=None, help='train_url')
    parser.add_argument('--data_url', type=str, default=None, help='data_url')
    parser.add_argument('--result_dir', type=str, default='./results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory name to save training logs')
    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --epoch
    assert args.epoch > 0, 'Totral number of epochs must be greater than zero'

    # --batch_size
    assert args.batch_size > 0, 'Batch size must be greater than zero'

    # --z_dim
    assert args.Z_dim > 0, 'Size of the noise vector must be greater than zero'

    return args


"""main function"""


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # Open New Tensorflow Session
    model = BicycleGAN
    #add
    # TMP_DATA_PATH = './' + args.dataset
    # TMP_RESULTS_PATH = '.' + args.result_dir
    # TMP_CHECKPOINT_PATH = './' + args.checkpoint_dir
    # TMP_LOGS_PATH = './' + args.log_dir

    # OBS_DATA_PATH = 'obs://bicyclegan/BicycleGAN2/' + args.dataset
    # OBS_RESULTS_PATH = 'obs://bicyclegan/BicycleGAN2/' + args.result_dir
    # OBS_CHECKPOINT_DIR = 'obs://bicyclegan/BicycleGAN2/' + args.checkpoint_dir
    # OBS_LOG_PATH = 'obs://bicyclegan/BicycleGAN2/' + args.log_dir
    # mox.file.make_dirs(TMP_DATA_PATH)
    # mox.file.make_dirs(TMP_RESULTS_PATH)
    # mox.file.make_dirs(TMP_CHECKPOINT_PATH)
    # mox.file.make_dirs(TMP_LOGS_PATH)
    # mox.file.copy_parallel(OBS_RESULTS_PATH, TMP_RESULTS_PATH)
    # mox.file.copy_parallel(OBS_DATA_PATH, TMP_DATA_PATH)
    # mox.file.copy_parallel(OBS_LOG_PATH, TMP_LOGS_PATH)
    # mox.file.copy_parallel(OBS_CHECKPOINT_DIR, TMP_CHECKPOINT_PATH)
    config = tf.ConfigProto(allow_soft_placement=True)
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Declare instance for GAN

        gan = None
        if args.gan_type == model.model_name:
            gan = model(sess,
                        epoch=args.epoch,
                        batch_size=args.batch_size,
                        Z_dim=args.Z_dim,
                        image_size=args.image_size,
                        dataset_name=args.dataset,
                        checkpoint_dir=args.checkpoint_dir,
                        result_dir=args.result_dir,
                        log_dir=args.log_dir)
        if gan is None:
            raise Exception("[!] There is no option for " + args.gan_type)

        # Build Tesnorflow Graph
        gan.build_model()

        # show network architecture
        # show_all_variables()

        # Launch the graph in a session
        gan.train()
        print(" [*] Training finished!")

        # visualize learned generator
        gan.test()
        print(" [*] Testing finished!")
        # mox.file.copy_parallel(TMP_RESULTS_PATH, OBS_RESULTS_PATH)
        # mox.file.copy_parallel(TMP_DATA_PATH, OBS_DATA_PATH)
        # mox.file.copy_parallel(TMP_LOGS_PATH, OBS_LOG_PATH)
        # mox.file.copy_parallel(TMP_CHECKPOINT_PATH, OBS_CHECKPOINT_DIR)


if __name__ == '__main__':
    main()
