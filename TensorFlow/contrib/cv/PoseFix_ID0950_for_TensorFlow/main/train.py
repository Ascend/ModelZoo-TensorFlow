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



import tensorflow as tf
import argparse
import numpy as np

from model import Model
from config import cfg
from tfflat.base import Trainer
from tfflat.utils import mem_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--train_epochs', type= int, dest='train_epochs',default=1)
    parser.add_argument('--batch_size', type= int, dest='batch_size',default=16)
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    args = parser.parse_args()

    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args
args = parse_args()

cfg.set_args(args.gpu_ids, args.continue_train)
cfg.set_attr("end_epoch", args.train_epochs)
cfg.set_attr("batch_size", args.batch_size)
trainer = Trainer(Model(), cfg)
trainer.train()



