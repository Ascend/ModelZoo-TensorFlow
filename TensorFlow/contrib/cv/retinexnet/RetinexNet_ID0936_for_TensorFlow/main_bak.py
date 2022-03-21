"""
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

from __future__ import print_function
from npu_bridge.npu_init import *
import os
import argparse
from glob import glob
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import moxing as mox

from PIL import Image
import tensorflow as tf

from model import LowlightEnhance
from utils import *


def lowlight_train(LowlightEnhance):
    """
    :param LowlightEnhance:
    :return:
    """
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    lr = args.start_lr * np.ones([args.epoch])
    lr[20:] = lr[0] / 10.0

    train_low_data = []
    train_high_data = []
    data_dir = args.modelarts_data_dir
    data_url = args.data_url
    newest_model = args.newest_model
    train_low_data_names = glob(data_dir + '/our485/low/*.png') + glob('./data/syn/low/*.png')
    train_low_data_names.sort()
    train_high_data_names = glob(data_dir + '/our485/high/*.png') + glob('./data/syn/high/*.png')
    train_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('[*] Number of training data: %d' % len(train_low_data_names))

    for idx in range(len(train_low_data_names)):
        low_im = load_images(train_low_data_names[idx])
        train_low_data.append(low_im)
        high_im = load_images(train_high_data_names[idx])
        train_high_data.append(high_im)

    eval_low_data = []
    eval_high_data = []

    eval_low_data_names = glob(data_dir + '/eval/low/*.*')

    for idx in range(len(eval_low_data_names)):
        eval_low_im = load_images(eval_low_data_names[idx])
        eval_low_data.append(eval_low_im)

    LowlightEnhance.train(train_low_data, train_high_data, eval_low_data, obs_result_dir, data_url, newest_model,
                           save_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, lr=lr,
                           sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Decom'),
                           eval_every_epoch=args.eval_every_epoch, train_phase="Decom")

    LowlightEnhance.train(train_low_data, train_high_data, eval_low_data, obs_result_dir, data_url, newest_model,
                           save_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, lr=lr,
                           sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Relight'),
                           eval_every_epoch=args.eval_every_epoch, train_phase="Relight")


def lowlight_test(LowlightEnhance):
    """
    :param LowlightEnhance:
    :return:
    """
    if args.test_dir is None:
        print("[!] please provide --test_dir")
        exit(0)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    newest_res = args.newest_res
    test_low_data_name = glob(os.path.join(args.test_dir) + '/*.*')
    test_low_data = []
    test_high_data = []
    for idx in range(len(test_low_data_name)):
        test_low_im = load_images(test_low_data_name[idx])
        test_low_data.append(test_low_im)

    LowlightEnhance.test(test_low_data, test_high_data, test_low_data_name, newest_res, save_dir=args.save_dir,
                          decom_flag=args.decom, model_path =args.model_path, save_data =save_data,
                          obs_result_dir=obs_result_dir)


def main(_):
    """

    :param _:
    :return:
    """
    if not args.use_gpu:
        print("[*] CPU\n")
        with tf.Session(config=config) as sess:
            model = LowlightEnhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)

    else:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        with tf.Session(config=config) as sess:
            model = LowlightEnhance(sess)
            if args.phase == 'train':
                lowlight_train(model)

            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
    parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
    parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.5, help="0 to 1, gpu memory usage")
    #parser.add_argument('--phase', dest='phase', default='train', help='train or test')
    parser.add_argument('--phase', dest='phase', default='train', help='train or test')

    parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of total epoches')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
    parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=20,
                        help='evaluating and saving checkpoints every #  epoch')
    parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
    parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='directory for evaluating outputs')

    parser.add_argument('--save_dir', dest='save_dir', default='./res', help='directory for testing outputs')  # res
    parser.add_argument('--test_dir', dest='test_dir', default='/cache/RetinexNet_data/test/low',
                        help='directory for testing inputs')
    parser.add_argument('--decom', dest='decom', default=0,
                        help='decom flag, 0 for enhanced results only and 1 for decomposition results')
    parser.add_argument('--model_path', dest='model_path', default='/cache/RetinexNet_data/model',
                        help='directory for testing model path')
    parser.add_argument('--newest_model', dest='newest_model', default='/newest_model',
                        help='directory for newset model saved path')
    parser.add_argument('--newest_res', dest='newest_res', default='/newest_res',
                        help='directory for newset resolt saved path')


    parser.add_argument("--train_url", type=str, default="./output")
    parser.add_argument("--data_url", type=str, default="./dataset")
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/RetinexNet_data")
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")

    args = parser.parse_args()
    config = parser.parse_args()
    mox.file.copy_parallel(src_url=config.data_url, dst_url=config.modelarts_data_dir)
    #if not os.path.exists(config.modelarts_result_dir):
    #    os.makedirs(config.modelarts_result_dir)
    obs_path = config.train_url
    obs_result_dir = obs_path + 'result'
    save_data = [config.ckpt_dir, config.sample_dir, config.save_dir]
    #path = config.modelarts_result_dir

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    tf.app.run()


