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


from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_config import DumpConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator, NPUEstimatorSpec
from npu_bridge.estimator.npu.npu_config import NPURunConfig
import tf_slim as slim
import tensorflow as tf
import time
import math
import os
os.system('pip install tf_slim')


class TrainingHook(tf.train.SessionRunHook):
    """A utility for displaying training information such as the loss, percent
    completed, estimated finish date and time."""

    def __init__(self, steps):
        self.steps = steps

        self.last_time = time.time()
        self.last_est = self.last_time

        self.eta_interval = int(math.ceil(0.1 * self.steps))
        self.current_interval = 0

    def before_run(self, run_context):
        """Function to collect the information"""
        graph = tf.get_default_graph()
        return tf.train.SessionRunArgs(
            {"loss": graph.get_collection("total_loss")[0],
             "loss_l1": graph.get_collection("loss_l1")[0],
             "loss_l2": graph.get_collection("loss_l2")[0],
             })

    def after_run(self, run_context, run_values):
        """Print the log after one step"""
        step = run_context.session.run(tf.train.get_global_step())
        now = time.time()

        if self.current_interval < self.eta_interval:
            self.duration = now - self.last_est
            self.current_interval += 1
        if step % self.eta_interval == 0:
            self.duration = now - self.last_est
            self.last_est = now

        eta_time = float(self.steps - step) / self.current_interval * \
            self.duration
        m, s = divmod(eta_time, 60)
        h, m = divmod(m, 60)
        eta = "%d:%02d:%02d" % (h, m, s)

        print("%.2f%% (%d/%d): loss: %.3e loss_l1:%.3e loss_l2:%.3e time: %.3f  end:%s (%s)" % (
            step * 100.0 / self.steps,
            step,
            self.steps,
            run_values.results["loss"],
            run_values.results["loss_l1"],
            run_values.results["loss_l2"],
            now - self.last_time,
            time.strftime("%a %d %H:%M:%S", time.localtime(
                time.time() + eta_time)),
            eta))

        self.last_time = now


def standard_model_fn(
        func, steps, run_config=None, sync_replicas=0, optimizer_fn=None):
    """Creates model_fn for tf.Estimator.

    Args:
      func: A model_fn with prototype model_fn(features, labels, mode, hparams).
      steps: Training steps.
      run_config: tf.estimatorRunConfig (usually passed in from TF_CONFIG).
      sync_replicas: The number of replicas used to compute gradient for
          synchronous training.
      optimizer_fn: The type of the optimizer. Default to Adam.

    Returns:
      model_fn for tf.estimator.Estimator.
    """

    def fn(features, labels, mode, params):
        """Returns model_fn for tf.estimator.Estimator."""

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        ret = func(features, labels, mode, params)

        tf.add_to_collection("total_loss", ret["loss"])
        tf.add_to_collection("loss_l1", ret["loss_l1"])
        tf.add_to_collection("loss_l2", ret["loss_l2"])

        train_op = None

        training_hooks = []
        if is_training:
            training_hooks.append(TrainingHook(steps))

            if optimizer_fn is None:
                optimizer = tf.train.AdamOptimizer(params.learning_rate)
            else:
                optimizer = optimizer_fn

            loss_scale_manager = ExponentialUpdateLossScaleManager(
                init_loss_scale=2**32, incr_every_n_steps=10000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
            optimizer = NPULossScaleOptimizer(optimizer, loss_scale_manager)

            if run_config is not None and run_config.num_worker_replicas > 1:
                sr = sync_replicas
                if sr <= 0:
                    sr = run_config.num_worker_replicas

                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer,
                    replicas_to_aggregate=sr,
                    total_num_replicas=run_config.num_worker_replicas)

                training_hooks.append(
                    optimizer.make_session_run_hook(
                        run_config.is_chief, num_tokens=run_config.num_worker_replicas))

            optimizer = tf.contrib.estimator.clip_gradients_by_norm(
                optimizer, 5)
            train_op = slim.learning.create_train_op(ret["loss"], optimizer)

        if "eval_metric_ops" not in ret:
            ret["eval_metric_ops"] = {}

        return NPUEstimatorSpec(
            mode=mode,
            predictions=ret["predictions"],
            loss=ret["loss"],
            train_op=train_op,
            eval_metric_ops=ret["eval_metric_ops"],
            training_hooks=training_hooks)
    return fn


def train_and_eval(
        model_dir,
        steps,
        batch_size,
        model_fn,
        create_input_fn,
        hparams,
        save_checkpoints_steps=2000,
        save_summary_steps=2000,
        sync_replicas=0):
    """Train the model with estimator"""

    config = tf.ConfigProto(
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    run_config = NPURunConfig(
        model_dir=model_dir,
        session_config=config,
        save_checkpoints_steps=save_checkpoints_steps,
        save_summary_steps=save_summary_steps,
        keep_checkpoint_max=1000)

    estimator = NPUEstimator(
        model_dir=model_dir,
        model_fn=standard_model_fn(
            model_fn,
            steps,
            run_config,
            sync_replicas=sync_replicas),
        params=hparams,
        config=run_config)

    estimator.train(
        input_fn=create_input_fn(batch_size=batch_size),
        max_steps=steps
    )
