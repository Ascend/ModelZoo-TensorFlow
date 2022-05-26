# Lint as: python2, python3
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
# ==============================================================================
"""An executor class for running model on TPUs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_config import ProfilingConfig

import collections
import os

from absl import logging

import numpy as np
import six
import tensorflow.compat.v1 as tf

from evaluation import coco_utils
from evaluation import factory
from hyperparameters import params_dict


def write_summary(logs, summary_writer, current_step):
  """Write out summaries of current training step for the checkpoint."""
  with tf.Graph().as_default():
    summaries = [
        tf.Summary.Value(tag=tag, simple_value=value)
        for tag, value in logs.items()
    ]
    tf_summary = tf.Summary(value=summaries)
    summary_writer.add_summary(tf_summary, current_step)


class TpuExecutor(object):
  """An executor class for running jobs on TPUs.

  Attributes:
    model_fn: The Model function for `tf.estimator.Estimator`.
    params: The (Retinanet) detecton parameter config.
    tpu_cluster_resolver: The `TPUClusterResolver` instance. If set, it will be
      directly passed to `tf.estimator.tpu.RunConfig`, otherwise, will need to
      construct a new `TPUClusterResolver` during construction.
  """

  def __init__(self,
               model_fn,
               params,
               tpu_cluster_resolver=None,
               keep_checkpoint_max=5):
    self._model_dir = params.model_dir
    self._params = params
    self._tpu_job_name = params.tpu_job_name
    self._evaluator = None
    self._tpu_cluster_resolver = tpu_cluster_resolver
    self._keep_checkpoint_max = keep_checkpoint_max

    input_partition_dims = None
    num_cores_per_replica = None

    if params.use_tpu or self._tpu_cluster_resolver:
      if not self._tpu_cluster_resolver:
        self._tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            params.platform.tpu,
            zone=params.platform.tpu_zone,
            project=params.platform.gcp_project)
      tpu_grpc_url = self._tpu_cluster_resolver.get_master()
      tf.Session.reset(tpu_grpc_url)

      # If the input image is transposed (from NHWC to HWCN), the partition
      # dimensions also need to be transposed the same way.
      def _maybe_transpose(input_partition_dims):
        if input_partition_dims and params.train.transpose_input:
          return [input_partition_dims[i] for i in [1, 2, 3, 0]]
        else:
          return input_partition_dims

      if params.train.input_partition_dims is not None:
        num_cores_per_replica = params.train.num_cores_per_replica
        input_partition_dims = params.train.input_partition_dims
        # Parse 'None' into None.
        input_partition_dims = [
            None if x == 'None' else _maybe_transpose(x)
            for x in input_partition_dims
        ]

      # Sets up config for TPUEstimator.
      tpu_config = tf.estimator.tpu.TPUConfig(
          params.train.iterations_per_loop,
          num_cores_per_replica=num_cores_per_replica,
          input_partition_dims=input_partition_dims,
          tpu_job_name=self._tpu_job_name,
          per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
          .PER_HOST_V2  # pylint: disable=line-too-long
      )

      run_config = tf.estimator.tpu.RunConfig(
          session_config=tf.ConfigProto(
              isolate_session_state=params.isolate_session_state),
          cluster=self._tpu_cluster_resolver,
          evaluation_master=params.platform.eval_master,
          model_dir=params.model_dir,
          log_step_count_steps=params.train.iterations_per_loop,
          tpu_config=tpu_config,
          keep_checkpoint_max=self._keep_checkpoint_max,
      )
      self._estimator = tf.estimator.tpu.TPUEstimator(
          model_fn=model_fn,
          use_tpu=False,
          train_batch_size=params.train.train_batch_size,
          eval_batch_size=params.eval.eval_batch_size,
          predict_batch_size=params.predict.predict_batch_size,
          config=npu_run_config_init(run_config=run_config),
          params=params.as_dict(), eval_on_tpu=False, export_to_tpu=False)
    else:
      model_params = params.as_dict()

      # Uses `train_batch_size` as the `batch_size` for GPU train.
      model_params.update({'batch_size': params.train.train_batch_size})

      gpu_devices = tf.config.experimental.list_physical_devices('GPU')
      tf.logging.info('gpu devices: %s', gpu_devices)
      devices = ['device:GPU:{}'.format(i) for i in range(len(gpu_devices))]
      strategy = npu_strategy.NPUStrategy()
      tf.logging.info('Number of devices: %s', strategy.num_replicas_in_sync)

      #profiling_dir = os.path.join(os.getcwd(), "npu_profiling")
      profiling_options = '{"output":"/tmp/npu_profiling",' \
                          '"training_trace":"on",' \
                          '"fp_point":"spinenet/batch_normalization/FusedBatchNormV3",' \
                          '"bp_point":"gradients/retinanet/box_net/box-3-3/FusedBatchNormV3_grad/FusedBatchNormGradV3"}'
      profiling_config = ProfilingConfig(enable_profiling=True,
                                         profiling_options=profiling_options)
      npu_run_config = NPURunConfig(
          profiling_config=profiling_config,
          train_distribute=strategy, model_dir=params.model_dir, log_step_count_steps=100)
      self._estimator = NPUEstimator(
           model_fn=model_fn, config=npu_run_config_init(npu_run_config), params=model_params)
      # run_config = tf.estimator.RunConfig(
      #     train_distribute=strategy, model_dir=params.model_dir, log_step_count_steps=1)
      # self._estimator = tf.estimator.Estimator(
      #     model_fn=model_fn, config=npu_run_config_init(run_config=npu_run_config), params=model_params)
 
  def train(self, input_fn, steps, hooks):
    """Training the model with training data and labels in input_fn."""
    self._estimator.train(input_fn=input_fn, max_steps=steps, hooks=hooks)

  def prepare_evaluation(self):
    """Preapre for evaluation."""
    eval_params = params_dict.ParamsDict(self._params.eval)
    if self._params.eval.type == 'box_and_mask':
      if (not self._params.eval.use_json_file or
          not self._params.eval.val_json_file):
        raise ValueError('If `eval.type` == `box_and_mask`, '
                         '`eval.val_json_file` is required.')
    if self._params.eval.use_json_file:
      val_json_file = os.path.join(self._params.model_dir,
                                   'eval_annotation_file.json')
      if self._params.eval.val_json_file:
        tf.gfile.Copy(
            self._params.eval.val_json_file, val_json_file, overwrite=True)
      else:
        coco_utils.scan_and_generator_annotation_file(
            self._params.eval.eval_file_pattern,
            self._params.eval.eval_samples,
            include_mask=False,
            annotation_file=val_json_file,
            dataset_type=self._params.eval.eval_dataset_type)
      eval_params.override({'val_json_file': val_json_file})
    self._evaluator = factory.evaluator_generator(eval_params)

  def evaluate(self, input_fn, eval_times, checkpoint_path=None):
    """Evaluating the model with data and labels in input_fn.

    Args:
      input_fn: Eval `input function` for tf.Estimator.
      eval_times: Int -  the number of times to evaluate.
      checkpoint_path: String - the checkpoint path to evaluate. If it is None,
        the latest checkpoint will be inferred from `model_dir` of `Estimator`.

    Returns:
      A dictionary as evaluation metrics.
    """
    if not checkpoint_path:
      checkpoint_path = self._estimator.latest_checkpoint()

    if self._params.eval.type == 'customized':
      metrics = self._estimator.evaluate(
          input_fn, steps=eval_times, checkpoint_path=checkpoint_path)
    else:
      if not self._evaluator:
        self.prepare_evaluation()
      if checkpoint_path:
        current_step = int(os.path.basename(checkpoint_path).split('-')[1])
      else:
        current_step = 0
      predictor = self._estimator.predict(
          input_fn=input_fn,
          checkpoint_path=checkpoint_path,
          yield_single_examples=False)
      losses = collections.defaultdict(lambda: 0.0)

      counter = 0
      try:
        while eval_times is None or counter < eval_times:
          outputs = six.next(predictor)
          predictions = {}
          groundtruths = {}
          for key, val in outputs.items():
            if key[0:5] == 'pred_':
              predictions[key[5::]] = val
            if key[0:3] == 'gt_':
              groundtruths[key[3::]] = val
            if key[0:5] == 'loss_':
              losses[key[5::]] += np.mean(val)
          self._evaluator.update(
              predictions,
              groundtruths=(None if self._params.eval.use_json_file
                            else groundtruths))
          counter = counter + 1
      except (tf.errors.OutOfRangeError, StopIteration):
        logging.info(
            'Evaluation reaches the end after running %d times.', counter)

      for key, val in outputs.items():
        if key[0:5] == 'loss_':
          losses[key[5::]] /= counter
      metrics = self._evaluator.evaluate()

      # Summary writer writes out eval metrics.
      output_dir = os.path.join(self._model_dir, 'eval')
      tf.gfile.MakeDirs(output_dir)
      summary_writer = tf.summary.FileWriter(output_dir)
      write_summary(metrics, summary_writer, current_step)
      write_summary(losses, summary_writer, current_step)
      summary_writer.close()

    logging.info('Eval result: %s', metrics)
    return metrics

  def predict(self, input_fn):
    return self._estimator.predict(input_fn=input_fn)

