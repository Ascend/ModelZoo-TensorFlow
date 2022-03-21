#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
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
#

"""
运行 BERT NER Server
#@Time    : 2019/1/26 21:00
# @Author  : MaCan (ma_cancan@163.com)
# @File    : run.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *


def start_server():
    from bert_base.server import BertServer
    from bert_base.server.helper import get_run_args

    args = get_run_args()
    print(args)
    server = BertServer(args)
    server.start()
    server.join()


def train_ner():
    import os
    from bert_base.train.train_helper import get_args_parser
    from bert_base.train.bert_lstm_ner import train

    args = get_args_parser()
    if True:
        import sys
        param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    print(args)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    train(args=args)


if __name__ == '__main__':
    """
    如果想训练，那么直接 指定参数跑，如果想启动服务，那么注释掉train,打开server即可
    """
    train_ner()
    #start_server()
