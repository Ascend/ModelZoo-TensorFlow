#!/usr/bin/env python
# coding=utf-8

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
# ============================================================================

from __future__ import division
from npu_bridge.npu_init import *
import yaml
import os
import argparse
import numpy as np
import logging

from utils.std_capturing import *
from model.monodepth2_learner import MonoDepth2Learner


def _cli_train(config, output_dir, args):
    with open(os.path.join(output_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    monodepth2_learner = MonoDepth2Learner(**config)
    monodepth2_learner.train(output_dir)

    print('Monodepth2 Training Done ...')


def _cli_test(config, output_dir, args):
    monodepth2_learner = MonoDepth2Learner(**config)
    monodepth2_learner.test(output_dir)

    print('Monodepth2 Test Done ...')


def _cli_eval(config, ckpt_name, args):
    monodepth2_learner = MonoDepth2Learner(**config)
    monodepth2_learner.eval(ckpt_name, args.eval_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Training command
    p_train = subparsers.add_parser('train')
    p_train.add_argument('config', type=str)
    p_train.add_argument('ckpt_name', type=str)
    p_train.set_defaults(func=_cli_train)

    # Test command
    p_test = subparsers.add_parser('test')
    p_test.add_argument('config', type=str)
    p_test.add_argument('ckpt_name', type=str)
    p_test.set_defaults(func=_cli_test)

    # Evaluate command
    p_eval = subparsers.add_parser('eval')
    p_eval.add_argument('config', type=str)
    p_eval.add_argument('ckpt_name', type=str)
    p_eval.add_argument('eval_type', type=str, default='depth', help='pose,depth')
    p_eval.set_defaults(func=_cli_eval)

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    output_dir = os.path.join(config['model']['root_dir'], args.ckpt_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with capture_outputs(os.path.join(output_dir, 'log')):
        logging.info('Running command {}'.format(args.command.upper()))
        args.func(config, output_dir, args)

