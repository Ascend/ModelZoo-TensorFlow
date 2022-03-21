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

"""Binary to train and evaluate one GAN configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
# pylint: disable=unused-import

from absl import app
from absl import flags
from absl import logging
from compare_gan import datasets
from compare_gan import runner_lib
# Import GAN types so that they can be used in Gin configs without module names.
from compare_gan.gans.modular_gan import ModularGAN
# Required import to configure core TF classes and functions.
import gin
import gin.tf.external_configurables
import tensorflow as tf
from npu_bridge.npu_init import *
#引入头文件


FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", None, "Where to store files.")
flags.DEFINE_string(
    "schedule", "train",
    "Schedule to run. Options: train, continuous_eval.")
flags.DEFINE_multi_string(
    "gin_config", [],
    "List of paths to the config files.")
flags.DEFINE_multi_string(
    "gin_bindings", [],
    "Newline separated list of Gin parameter bindings.")
flags.DEFINE_string(
    "score_filename", "scores.csv",
    "Name of the CSV file with evaluation results model_dir.")

flags.DEFINE_integer(
    "num_eval_averaging_runs", 1,
    "How many times to average FID and IS")
flags.DEFINE_integer(
    "eval_every_steps", 100,
    "Evaluate only checkpoints whose step is divisible by this integer")

flags.DEFINE_bool("use_tpu", None, "Whether running on TPU or not.")


def _get_cluster():
  if not FLAGS.use_tpu:  # pylint: disable=unreachable
    return None
  if "TPU_NAME" not in os.environ:
    raise ValueError("Could not find a TPU. Set TPU_NAME.")
  return tf.contrib.cluster_resolver.TPUClusterResolver(
      tpu=os.environ["TPU_NAME"],
      zone=os.environ.get("TPU_ZONE", None))


@gin.configurable("run_config")
def _get_run_config(tf_random_seed=None,
                    single_core=False,
                    iterations_per_loop=100,
                    save_checkpoints_steps=1000,
                    keep_checkpoint_max=1000):
  """Return `RunConfig` for TPUs."""
  from npu_bridge.estimator.npu.npu_config import NPURunConfig
  from npu_bridge.estimator import npu_ops
  config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
  return NPURunConfig(
        model_dir=FLAGS.model_dir,
        iterations_per_loop=iterations_per_loop,
        tf_random_seed=tf_random_seed,
        save_checkpoints_steps=save_checkpoints_steps,
        keep_checkpoint_max=keep_checkpoint_max,
        session_config=config)




def _get_task_manager():
  """Returns a TaskManager for this experiment."""
  score_file = os.path.join(FLAGS.model_dir, FLAGS.score_filename)
  return runner_lib.TaskManagerWithCsvResults(
      model_dir=FLAGS.model_dir, score_file=score_file)


def main(unused_argv):
  logging.info("Gin config: %s\nGin bindings: %s",
               FLAGS.gin_config, FLAGS.gin_bindings)
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)


  if FLAGS.use_tpu is None:
    FLAGS.use_tpu = bool(os.environ.get("TPU_NAME", ""))
    if FLAGS.use_tpu:
      logging.info("Found TPU %s.", os.environ["TPU_NAME"])
  run_config = _get_run_config()
  task_manager = _get_task_manager()
  options = runner_lib.get_options_dict()
  runner_lib.run_with_schedule(
      schedule=FLAGS.schedule,
      run_config=run_config,
      task_manager=task_manager,
      options=options,
      use_tpu=FLAGS.use_tpu,
      num_eval_averaging_runs=FLAGS.num_eval_averaging_runs,
      eval_every_steps=FLAGS.eval_every_steps)
  logging.info("I\"m done with my work, ciao!")


if __name__ == "__main__":
  flags.mark_flag_as_required("model_dir")
  app.run(main)
