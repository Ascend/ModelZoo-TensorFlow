# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
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
"""Train ConvNet Mean Teacher on CIFAR-10 training set and evaluate against a validation set

This runner converges quickly to a fairly good accuracy.
On the other hand, the runner experiments/cifar10_final_eval.py
contains the hyperparameters used in the paper, and converges
much more slowly but possibly to a slightly better accuracy.
"""
from npu_bridge.npu_init import *
import argparse
import logging

from experiments.run_context import RunContext
from datasets import Cifar10ZCA
from mean_teacher.model import Model
from mean_teacher import minibatching

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('main')
parser = argparse.ArgumentParser()
parser.add_argument('--data_path',type=str, default='data',help='The path of dataset')
parser.add_argument('--n_labeled',type=int, default=4000,help='The num of labeled images')
parser.add_argument('--training_length',type=int, default=40000,help='The steps o training')
parser.add_argument('--output_path',type=str, default='output',help='The path of output')

#running function
def run(data_seed, args):
    n_labeled = args.n_labeled
    data_path = args.data_path
    output_path = args.output_path
    model = Model(RunContext(__file__, 0, output_path))
    model['flip_horizontally'] = True
    model['normalize_input'] = False  # Keep ZCA information
    model['rampdown_length'] = 0
    model['rampup_length'] = 5000
    model['training_length'] = args.training_length
    model['max_consistency_cost'] = 50.0

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    cifar = Cifar10ZCA(data_seed, n_labeled, data_path)
    training_batches = minibatching.training_batches(cifar.training, n_labeled_per_batch=50)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(cifar.evaluation)

    model.train(training_batches, evaluation_batches_fn)

if __name__ == "__main__":
    args = parser.parse_args()
    run(0,args)

