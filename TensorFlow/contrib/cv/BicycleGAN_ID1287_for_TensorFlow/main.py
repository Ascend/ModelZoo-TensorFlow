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
import argparse
import os
import tensorflow as tf
from load_data import load_images
from model import BicycleGAN
from folder import check_folder
from eval import eval_tf
from npu_bridge.npu_init import *
from tensorflow_core.core.protobuf.rewriter_config_pb2 import RewriterConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Z_dim', type=int, default=8, help='Size of latent vector')
    parser.add_argument('--reconst_coeff', type=float, default=10, help='Reconstruction Coefficient')
    parser.add_argument('--latent_coeff', type=float, default=0.5, help='Latent Coefficient')
    parser.add_argument('--kl_coeff', type=float, default=0.01, help='KL Coefficient')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning Rate')
    parser.add_argument('--image_size', type=int, default=256, help='Image Size')
    parser.add_argument('--batch_size', type=int, default=1, help='number of images in one minibatch')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--data_path', type=str, default='', help='Datasets location')
    parser.add_argument('--output_path', type=str, default='', help='Output location')
    return check_args(parser.parse_args())


def check_args(args):
    # --result_dir
    check_folder(args.output_path)

    # --epoch
    assert args.epoch > 0, 'Totral number of epochs must be greater than zero'

    # --batch_size
    # Due to the limit of the network,the batch_size must be set to 1 currently
    assert args.batch_size > 0, 'Batch size must be greater than zero'

    # --z_dim
    assert args.Z_dim > 0, 'Size of the noise vector must be greater than zero'

    return args


def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # NPU config
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

    # Open New Tensorflow Session
    model = BicycleGAN
    with tf.Session(config=config) as sess:
        gan = model(sess=sess, output_path=args.output_path,
                    epoch=args.epoch, batch_size=args.batch_size,
                    image_size=args.image_size, Z_dim=args.Z_dim,
                    learning_rate=args.learning_rate, reconst_coeff=args.reconst_coeff,
                    latent_coeff=args.latent_coeff, kl_coeff=args.kl_coeff)

        train_A, train_B, test_A, test_B = load_images(args.data_path, args.image_size)
        assert len(test_A) == len(test_B)
        assert len(train_A) == len(train_B)

        gan.train(train_A=train_A, train_B=train_B)
        print(" [*] Training finished!")

        gan.test(test_A=test_A)
        print(" [*] Testing finished!")

        path = os.path.join(args.output_path, "results", "test_results")
        eval_tf(path)


if __name__ == '__main__':
    main()
