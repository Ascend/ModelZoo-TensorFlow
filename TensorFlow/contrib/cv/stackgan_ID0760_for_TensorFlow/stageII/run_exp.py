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

import dateutil.tz
import datetime
import argparse
import pprint

import sys
sys.path.append('misc')
sys.path.append('stageII')

from datasets import TextDataset
from utils import mkdir_p
from config import cfg, cfg_from_file
from model import CondGAN
from trainer import CondGANTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=-1, type=int)
    parser.add_argument('--bin', dest='bin', help='run on 310', action="store_true",default=False)
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    print('Using config:')
    pprint.pprint(cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    #timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    datadir = 'Data/%s' % cfg.DATASET_NAME
    dataset = TextDataset(datadir,  cfg.EMBEDDING_TYPE, 4)
    filename_test = '%s/test' % (datadir)
    dataset.test = dataset.get_data(filename_test)
    if cfg.TRAIN.FLAG:
        filename_train = '%s/train' % (datadir)
        dataset.train = dataset.get_data(filename_train)
        ckt_logs_dir = "ckt_logs/%s/%s" % (cfg.DATASET_NAME, cfg.CONFIG_NAME) #看护需要，不便使用日期，所以去除
        mkdir_p(ckt_logs_dir)
    else:
        s_tmp = cfg.TRAIN.PRETRAINED_MODEL
        ckt_logs_dir = s_tmp[:s_tmp.find('.ckpt')]
    model = CondGAN(lr_imsize=int(dataset.image_shape[0] / dataset.hr_lr_ratio), hr_lr_ratio=dataset.hr_lr_ratio)

    algo = CondGANTrainer(model=model, dataset=dataset, ckt_logs_dir=ckt_logs_dir)
    if args.bin:
        print("##############")
        algo.evaluate_bin()
    else:
        if cfg.TRAIN.FLAG:
            algo.train()
        else:

            ''' For every input text embedding/sentence in the
            training and test datasets, generate cfg.TRAIN.NUM_COPY
            images with randomness from noise z and conditioning augmentation.'''

            algo.evaluate()


