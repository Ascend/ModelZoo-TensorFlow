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
Train 3D U-Net network, for prostate MRI scans.

Ideas taken from:
https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/vision

and

https://github.com/tensorflow/models/blob/master/samples/core/
get_started/custom_estimator.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import os
import sys
import pickle
import argparse

import tensorflow as tf

from src.model_fn import model_fn
from src.input_fn import input_fn
from src.utils import Params, set_logger


def arg_parser(args):
    """
    Define cmd line help for main.
    """
    
    parser_desc = "Train, eval, predict 3D U-Net model."
    parser = argparse.ArgumentParser(description=parser_desc)
    
    parser.add_argument(
        '-model_dir', 
        default='../models/base_model',
        required=True,
        help="Experiment directory containing params.json"
    )
    parser.add_argument(
        '-mode', 
        default='train_eval',
        help="One of train, train_eval, eval, predict."
    )

    parser.add_argument(
        '-pred_ix',
        nargs='+',
        type=int,
        default=[1],
        help="Space separated list of indices of patients to predict."
    )
    # modify for NPU start
    parser.add_argument(
        '-precision_mode',
        default='allow_fp32_to_fp16',
        help="Precision mode. Default is allow_fp32_to_fp16"
    )
    parser.add_argument(
        '-step',
        default=7000,
        type=int,
        help="train step"
    )
    parser.add_argument(
        '-train_path',
        default='data/processed/train_dataset_resizes.pckl',
        help="train_dataset_path"
    )
    parser.add_argument(
        '-test_path',
        default='data/processed/test_dataset.pckl',
        help="test_dataset_path"
    )
    # modify for NPU end
    # parse input params from cmd line
    try:
        return parser.parse_args(args)
    except:
        parser.print_help()
        sys.exit(0)


def main(argv):
    """
    Main driver/runner of 3D U-Net model.
    """
    
    # -------------------------------------------------------------------------
    # setup
    # -------------------------------------------------------------------------

    # set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(42)

    # load the parameters from model's json file as a dict
    args = arg_parser(argv)
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path).dict
    
    ## NPU modify begin
    params['max_train_steps'] = args.step
    params['train_dataset_path'] = args.train_path
    params['test_dataset_path'] = args.test_path
    ## NPU modify end
    
    # check mode
    modes = ['train', 'train_eval', 'eval', 'predict']
    assert args.mode in modes, "mode has to be one of %s" % ','.join(modes) 
    
    # create logger, add loss and IOU to logging
    logger = set_logger(os.path.join(args.model_dir, 'train.log'))
    
    # -------------------------------------------------------------------------
    # create model
    # -------------------------------------------------------------------------
    #Modify for NPU start
    session_config = tf.ConfigProto()
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    if args.precision_mode == "allow_mix_precision":
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

    run_config = tf.estimator.RunConfig(
        session_config=session_config,
        log_step_count_steps=params['display_steps']
    )

    if args.precision_mode == "allow_mix_precision":
        params['precision_mode'] = "allow_mix_precision"

    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
        params=params,
        config=npu_run_config_init(run_config=run_config))


    # model = tf.estimator.Estimator(
    #     model_fn=model_fn,
    #     model_dir=args.model_dir,
    #     params=params,
    #     config=npu_run_config_init(run_config=tf.estimator.RunConfig(
    #         log_step_count_steps=params['display_steps']
    #     ))
    # )

    # Modify for NPU end

    # -------------------------------------------------------------------------
    # train
    # -------------------------------------------------------------------------
    
    if args.mode in ['train_eval', 'train']:
        model.train(
            input_fn=lambda: input_fn(True, params),
            max_steps=params['max_train_steps']
        , hooks=npu_hooks_append())
    
    # -------------------------------------------------------------------------
    # evaluate
    # -------------------------------------------------------------------------
    
    if args.mode in ['train_eval', 'eval']:
        model.evaluate(input_fn=lambda: input_fn(False, params))
    
    # -------------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------------
    
    if args.mode == 'predict':
        predictions = model.predict(input_fn=lambda: input_fn(False, params))

        # extract predictions, only save predicted classes not probs
        to_save = dict()
        for i, y_pred in enumerate(predictions):
            if i in args.pred_ix:
                logger.info('Predicting patient: %d.' % i)
                to_save[i] = y_pred
        
        # save them with pickle to model dir
        pred_file = os.path.join(args.model_dir, 'preds.npy')
        pickle.dump(to_save, open(pred_file,"wb"))
        logger.info('Predictions saved to: %s.' % pred_file)


if __name__ == '__main__':
    main(sys.argv[1:])
