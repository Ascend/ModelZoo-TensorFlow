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
"""Trainer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import os
import argparse
import importlib

import tensorflow as tf

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset',
      help='Dataset name',
      default=None,
      type=str,
  )
  parser.add_argument(
      '--model',
      help='Model name',
      default=None,
      type=str,
  )
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True)

  # Experiment arguments
  parser.add_argument(
      '--save-checkpoints-steps',
      help='Number of steps to save checkpoint',
      default=1000,
      type=int)
  parser.add_argument(
      '--train-steps',
      help='Number of steps to run training totally',
      default=None,
      type=int)
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evaluation for at each checkpoint',
      default=None,
      type=int)
  parser.add_argument(
      '--save-summary-steps',
      help='Number of steps to save summary',
      default=100,
      type=int)
  parser.add_argument(
      '--random-seed',
      help='Random seed for TensorFlow',
      default=None,
      type=int)
  # Performance tuning parameters
  parser.add_argument(
      '--allow-growth',
      help='Whether to enable allow_growth in GPU_Options',
      default=False,
      action='store_true')
  parser.add_argument(
      '--xla',
      help='Whether to enable XLA auto-jit compilation',
      default=False,
      action='store_true')
  parser.add_argument(
      '--save-profiling-steps',
      help='Number of steps to save profiling',
      default=None,
      type=int)
  # Argument to turn on all logging
  parser.add_argument(
      '--verbosity',
      choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
      default='INFO',
      help='Set logging verbosity')

  # Parse arguments
  args, _ = parser.parse_known_args()
  dataset_module = importlib.import_module('datasets.' + args.dataset
                                           if args.dataset else 'datasets')
  dataset_module.update_argparser(parser)
  model_module = importlib.import_module('models.' + args.model
                                         if args.model else 'models')
  model_module.update_argparser(parser)
  hparams = parser.parse_args()
  print(hparams)

  # Set python level verbosity
  tf.logging.set_verbosity(hparams.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[hparams.verbosity] / 10)

  # Run the training job
  model_fn = getattr(model_module, 'model_fn')
  input_fn = getattr(dataset_module, 'input_fn')
  train_input_fn = lambda: input_fn(
      mode=tf.estimator.ModeKeys.TRAIN,
      params=hparams,
  )
  eval_input_fn = lambda: input_fn(
      mode=tf.estimator.ModeKeys.EVAL,
      params=hparams,
  )
  predict_input_fn = getattr(dataset_module, 'predict_input_fn', None)

  #session_config = tf.ConfigProto()
  ####################NPU_modify start####################
  config = tf.ConfigProto()
  custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer"
  custom_op.parameter_map["use_off_line"].b = True
  custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

  cur_path = os.path.dirname(os.path.abspath(__file__))
  blacklist_file = os.path.join(cur_path, "ops_info.json")
  custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes(blacklist_file)

  config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  session_config = npu_config_proto(config_proto=config)
  ####################NPU_modify end####################

  session_config.gpu_options.allow_growth = hparams.allow_growth
  if hparams.xla:
    session_config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)
  run_config = tf.estimator.RunConfig(
      model_dir=hparams.job_dir,
      tf_random_seed=hparams.random_seed,
      save_summary_steps=hparams.save_summary_steps,
      save_checkpoints_steps=hparams.save_checkpoints_steps,
      session_config=session_config,
  )
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=npu_run_config_init(run_config=run_config),
      params=hparams,
  )
  hooks = []
  if hparams.save_profiling_steps:
    hooks.append(
        tf.train.ProfilerHook(
            save_steps=hparams.save_profiling_steps,
            output_dir=hparams.job_dir,
        ))
  train_spec = tf.estimator.TrainSpec(
      input_fn=train_input_fn,
      max_steps=hparams.train_steps,
      hooks=npu_hooks_append(hooks_list=hooks),
  )
  eval_spec = tf.estimator.EvalSpec(
      input_fn=eval_input_fn,
      steps=hparams.eval_steps,
      exporters=tf.estimator.LatestExporter(
          name='Servo', serving_input_receiver_fn=predict_input_fn)
      if predict_input_fn else None, hooks=npu_hooks_append(),
  )
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
