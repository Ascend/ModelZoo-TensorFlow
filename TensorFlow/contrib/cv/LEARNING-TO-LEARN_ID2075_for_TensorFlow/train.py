# Copyright 2016 Google Inc.
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
"""Learning 2 Learn training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
import os

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms
from tensorflow.python.training import monitored_session
from tensorflow.train import ChiefSessionCreator

#import moxing as mox
import meta
import util

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS
flags.DEFINE_string("save_path", "./mnist", "Path for saved meta-optimizer.")
flags.DEFINE_integer("num_epochs", 10000, "Number of training epochs.")
flags.DEFINE_integer("log_period", 50, "Log period.")
flags.DEFINE_integer("evaluation_period", 100, "Evaluation period.")
flags.DEFINE_integer("evaluation_epochs", 20, "Number of evaluation epochs.")

flags.DEFINE_string("problem", "mnist", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.")
flags.DEFINE_integer("unroll_length", 20, "Meta-optimizer unroll length.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_boolean("second_derivatives", False, "Use second derivatives.")
flags.DEFINE_string("obs_dir", "/modelart-zs/", "Use second derivatives.")

def main(_):
  # Configuration.
  num_unrolls = FLAGS.num_steps // FLAGS.unroll_length

  if FLAGS.save_path is not None:
    if os.path.exists(FLAGS.save_path):

      raise ValueError("Folder {} already exists".format(FLAGS.save_path))

    else:
      os.mkdir(FLAGS.save_path)

  # Problem.
  problem, net_config, net_assignments = util.get_config(FLAGS.problem)

  # Optimizer setup.
  optimizer = meta.MetaOptimizer(**net_config)
  minimize = optimizer.meta_minimize(
      problem, FLAGS.unroll_length,
      learning_rate=FLAGS.learning_rate,
      net_assignments=net_assignments,
      second_derivatives=FLAGS.second_derivatives)
  step, update, reset, cost_op, _ = minimize

  config = tf.ConfigProto()
  custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer"
  custom_op.parameter_map["profiling_mode"].b = True

  custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
    '{"output":"/home/ma-user/modelarts/outputs/train_url_0/","task_trace":"on"}')
  #custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")
  config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # ??????????????????
  config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # ??????????????????

  #MonitoredSession = monitored_session.SingularMonitoredSession(config=config)
  with ms.MonitoredSession(session_creator=ChiefSessionCreator(config=config)) as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()

    best_evaluation = float("inf")
    total_time = 0
    total_cost = 0
    for e in xrange(FLAGS.num_epochs):
      # Training.
      time, cost = util.run_epoch(sess, cost_op, [update, step], reset,
                                  num_unrolls)
      total_time += time
      total_cost += cost

      # Logging.
      if (e + 1) % FLAGS.log_period == 0:
        util.print_stats("Epoch {}".format(e + 1), total_cost, total_time,
                         FLAGS.log_period)
        total_time = 0
        total_cost = 0

      # Evaluation.
      if (e + 1) % FLAGS.evaluation_period == 0:
        eval_cost = 0
        eval_time = 0
        for _ in xrange(FLAGS.evaluation_epochs):
          time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                      num_unrolls)
          eval_time += time
          eval_cost += cost

        util.print_stats("EVALUATION", eval_cost, eval_time,
                         FLAGS.evaluation_epochs)



        if FLAGS.save_path is not None and eval_cost < best_evaluation:
          print("Removing previously saved meta-optimizer")
          for f in os.listdir(FLAGS.save_path):
            os.remove(os.path.join(FLAGS.save_path, f))
          print("Saving meta-optimizer to {}".format(FLAGS.save_path))
          optimizer.save(sess, FLAGS.save_path)
          best_evaluation = eval_cost
        '''

        obs_model = os.path.join(FLAGS.obs_dir, 'model')
        if not mox.file.exists(obs_model):
            mox.file.make_dirs(obs_model)
        mox.file.copy_parallel(FLAGS.save_path, obs_model)
        '''


if __name__ == "__main__":
  tf.app.run()
