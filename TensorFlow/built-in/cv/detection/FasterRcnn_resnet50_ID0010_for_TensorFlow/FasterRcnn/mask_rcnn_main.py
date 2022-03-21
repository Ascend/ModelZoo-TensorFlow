# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Training script for Mask-RCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

import os
import re
import sys
sys.path.insert(0, 'tpu/models')
from hyperparameters import common_hparams_flags
from hyperparameters import common_tpu_flags
from hyperparameters import flags_to_params
from hyperparameters import params_dict
import dataloader
import distributed_executer
import mask_rcnn_model
from configs import mask_rcnn_config
from hccl.split.api import set_split_strategy_by_size
set_split_strategy_by_size([70, 29, 1], 'hccl_world_group')

common_tpu_flags.define_common_tpu_flags()
common_hparams_flags.define_common_hparams_flags()

# set random seed
rnd_seed = 0
tf.set_random_seed(rnd_seed)
import numpy as np
np.random.seed(rnd_seed)
import random
random.seed(rnd_seed)
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

flags.DEFINE_string(
    'distribution_strategy',
    default='multi_worker_gpu',
    help='Distribution strategy or estimator type to use. One of'
    '"multi_worker_gpu"|"tpu".')

# Parameters for MultiWorkerMirroredStrategy
flags.DEFINE_string(
    'worker_hosts',
    default=None,
    help='Comma-separated list of worker ip:port pairs for running '
    'multi-worker models with distribution strategy.  The user would '
    'start the program on each host with identical value for this flag.')
flags.DEFINE_integer(
    'task_index', 0, 'If multi-worker training, the task_index of this worker.')
flags.DEFINE_integer(
    'num_gpus',
    default=0,
    help='Number of gpus when using collective all reduce strategy.')
flags.DEFINE_integer(
    'worker_replicas',
    default=0,
    help='Number of workers when using collective all reduce strategy.')

# TPUEstimator parameters
flags.DEFINE_integer(
    'num_cores', default=None, help='Number of TPU cores for training')
flags.DEFINE_multi_integer(
    'input_partition_dims', None,
    'A list that describes the partition dims for all the tensors.')
flags.DEFINE_bool(
    'transpose_input',
    default=None,
    help='Use TPU double transpose optimization')
flags.DEFINE_string(
    'tpu_job_name', None,
    'Name of TPU worker binary. Only necessary if job name is changed from'
    ' default tpu_worker.')

# Model specific paramenters
flags.DEFINE_string('mode', 'train',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')


FLAGS = flags.FLAGS


def run_executer(model_params, train_input_fn=None, eval_input_fn=None):
  """Runs Mask RCNN model on distribution strategy defined by the user."""

  if FLAGS.distribution_strategy == 'multi_worker_gpu':
    executer = distributed_executer.MultiWorkerExecuter(
        FLAGS, model_params, mask_rcnn_model.mask_rcnn_model_fn)
  else:
    executer = distributed_executer.TPUEstimatorExecuter(
        FLAGS, model_params, mask_rcnn_model.mask_rcnn_model_fn)

  if FLAGS.mode == 'train':
    executer.train(train_input_fn, FLAGS.eval_after_training, eval_input_fn)
  elif FLAGS.mode == 'eval':
    executer.eval(eval_input_fn)
  elif FLAGS.mode == 'train_and_eval':
    executer.train_and_eval(train_input_fn, eval_input_fn)
  else:
    raise ValueError('Mode must be one of `train`, `eval`, or `train_and_eval`')


def main(argv):
  del argv  # Unused.


  print("\n================ FLAGS:")
  for item in FLAGS:
    print(item, ':', FLAGS[item].value)

  # Configure parameters.
  params = params_dict.ParamsDict(
      mask_rcnn_config.MASK_RCNN_CFG, mask_rcnn_config.MASK_RCNN_RESTRICTIONS)
  #params = params_dict.override_params_dict(
  #    params, FLAGS.config_file, is_strict=True)

  # if there exists <model_dir>/params.yaml, then overwrite params
  resume_params_file = FLAGS.model_dir + "/params.yaml"
  if os.path.exists(resume_params_file):
    tf.logging.info(f"################ Resume with param file : {resume_params_file}")
    params.override({"model_dir": FLAGS.model_dir}, is_strict=False)
    params = params_dict.override_params_dict(params, resume_params_file, is_strict=True)

  # if there exists 1 or more config_file, overwrite params one by one
  # separate file names by , or ;
  if FLAGS.config_file :
    config_files = re.split(",|;", FLAGS.config_file)
    tf.logging.info(f"config_files : {config_files}")
    for config_file in config_files:
      params = params_dict.override_params_dict(params, config_file, is_strict=True)

  # overwrite params with --params_override=<name>=<value>,<name>=<value> in command line
  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)
  # overwrite params with --<name>=<value> in command line
  params = flags_to_params.override_params_from_input_flags(params, FLAGS)
  # new param: num_gpus
  if FLAGS.num_gpus:
    params.override({"num_gpus": FLAGS.num_gpus}, is_strict=False)

  params.validate()
  params.lock()

  import json
  beautiful_cfg = json.dumps(params.as_dict(), indent=2, ensure_ascii=False)
  print("\n================ params:")
  print(beautiful_cfg)

  # write logging into file
  log_file = FLAGS.mode + "_logging.log"
  log_dir = FLAGS.model_dir + "/log/"
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  logging.get_absl_handler().use_absl_log_file(log_file, log_dir)

  # Check data path
  train_input_fn = None
  eval_input_fn = None
  print(params.training_file_pattern)
  if (FLAGS.mode in ('train', 'train_and_eval') and
      not params.training_file_pattern):
    raise RuntimeError('You must specify `training_file_pattern` for training.')
  if FLAGS.mode in ('eval', 'train_and_eval'):
    if not params.validation_file_pattern:
      raise RuntimeError('You must specify `validation_file_pattern` '
                         'for evaluation.')
    if not params.val_json_file and not params.include_groundtruth_in_features:
      raise RuntimeError('You must specify `val_json_file` or '
                         'include_groundtruth_in_features=True for evaluation.')

  if FLAGS.mode in ('train', 'train_and_eval'):
    train_input_fn = dataloader.InputReader(
        params.training_file_pattern,
        mode=tf.estimator.ModeKeys.TRAIN,
        use_fake_data=FLAGS.use_fake_data,
        use_instance_mask=params.include_mask)
  if (FLAGS.mode in ('eval', 'train_and_eval') or
      (FLAGS.mode == 'train' and FLAGS.eval_after_training)):
    eval_input_fn = dataloader.InputReader(
        params.validation_file_pattern,
        mode=tf.estimator.ModeKeys.PREDICT,
        num_examples=params.eval_samples,
        use_instance_mask=params.include_mask)

  run_executer(params, train_input_fn, eval_input_fn)


if __name__ == '__main__':
  tf.disable_eager_execution()
  logging.set_verbosity(logging.INFO)
  tf.app.run(main)
