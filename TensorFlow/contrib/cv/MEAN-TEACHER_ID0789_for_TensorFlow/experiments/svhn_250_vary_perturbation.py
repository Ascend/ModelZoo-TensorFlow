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
"""Vary different perturbations on 250-label SVHN for the NIPS paper"""
from npu_bridge.npu_init import *

import logging
import sys

from .run_context import RunContext
import tensorflow as tf

from datasets import SVHN
from mean_teacher.model import Model
from mean_teacher import minibatching


LOG = logging.getLogger('main')


def all_parameters():
    n_runs = 4
    for data_seed in range(1000, 1000 + n_runs):
        for dropout in [0.5, 0.]:
            for input_noise in [0.15, 0.]:
                for augmentation in [True, False]:
                    yield {
                        'data_seed': data_seed,
                        'dropout': dropout,
                        'input_noise': input_noise,
                        'augmentation': augmentation,
                    }


def parameters():
    for idx, params in enumerate(all_parameters()):
        if idx >= 6:
            yield params


def model_hyperparameters(model_type, n_labeled, n_extra_unlabeled):
    assert model_type in ['mean_teacher', 'pi']
    training_length = {
        0: 180000,
        100000: 400000,
        500000: 600000,
    }
    if n_labeled == 'all':
        return {
            'training_length': training_length[n_extra_unlabeled],
            'n_labeled_per_batch': 100,
            'max_consistency_cost': 100.0,
            'apply_consistency_to_labeled': True,
            'ema_consistency': model_type == 'mean_teacher'
        }
    elif isinstance(n_labeled, int):
        return {
            'training_length': training_length[n_extra_unlabeled],
            'n_labeled_per_batch': 1,
            'max_consistency_cost': 1.0,
            'apply_consistency_to_labeled': False,
            'ema_consistency': model_type == 'mean_teacher'
        }
    else:
        msg = "Unexpected combination: {model_type}, {n_labeled}, {n_extra_unlabeled}"
        assert False, msg.format(locals())


def run(data_seed, dropout, input_noise, augmentation,
        test_phase=False, n_labeled=250, n_extra_unlabeled=0, model_type='mean_teacher'):
    minibatch_size = 100
    hyperparams = model_hyperparameters(model_type, n_labeled, n_extra_unlabeled)

    tf.reset_default_graph()
    model = Model(RunContext(__file__, data_seed))

    svhn = SVHN(n_labeled=n_labeled,
                n_extra_unlabeled=n_extra_unlabeled,
                data_seed=data_seed,
                test_phase=test_phase)

    model['ema_consistency'] = hyperparams['ema_consistency']
    model['max_consistency_cost'] = hyperparams['max_consistency_cost']
    model['apply_consistency_to_labeled'] = hyperparams['apply_consistency_to_labeled']
    model['training_length'] = hyperparams['training_length']
    model['student_dropout_probability'] = dropout
    model['teacher_dropout_probability'] = dropout
    model['input_noise'] = input_noise
    model['translate'] = augmentation

    training_batches = minibatching.training_batches(svhn.training,
                                                     minibatch_size,
                                                     hyperparams['n_labeled_per_batch'])
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(svhn.evaluation,
                                                                    minibatch_size)

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)

