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
"""Learning 2 Learn evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf
import numpy as np

from npu_bridge.npu_init import *
from tensorflow.contrib.learn.python.learn import monitored_session as ms


import matplotlib.pyplot as plt
import meta
import util
from tensorflow.train import ChiefSessionCreator
flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_string("optimizer", "L2L", "Optimizer.")
flags.DEFINE_string("path","./npucheck/" , "Path to saved meta-optimizer network.")
flags.DEFINE_integer("num_epochs", 100, "Number of evaluation epochs.")
flags.DEFINE_integer("seed", None, "Seed for TensorFlow's RNG.")

flags.DEFINE_string("problem", "mnist", "Type of problem.")
flags.DEFINE_integer("num_steps", 2,
                     "Number of optimization steps per epoch.")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")


def print_stats(header, total_error, total_time, n):
  """Prints experiment statistics."""
  LMFE = np.log10(total_error / n)
  MET = total_time / n
  print(header)
  print("Log Mean Final Error: {:.2f}".format(np.log10(total_error / n)))
  print("Mean epoch time: {:.2f} s".format(total_time / n))
  return LMFE,MET

def main(_):
  # Configuration.
  num_unrolls = FLAGS.num_steps

  if FLAGS.seed:
    tf.set_random_seed(FLAGS.seed)

  # Problem.
  problem, net_config, net_assignments = util.get_config(FLAGS.problem,
                                                         FLAGS.path)
  costlist = []
  x_list = []
  # Optimizer setup.
  if FLAGS.optimizer == "Adam":
    cost_op = problem()
    problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    problem_reset = tf.variables_initializer(problem_vars)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
    update = optimizer.minimize(cost_op)
    reset = [problem_reset, optimizer_reset]
  elif FLAGS.optimizer == "L2L":
    if FLAGS.path is None:
      logging.warning("Evaluating untrained L2L optimizer")
    optimizer = meta.MetaOptimizer(**net_config)
    meta_loss = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)
    _, update, reset, cost_op, _ = meta_loss
  else:
    raise ValueError("{} is not a valid optimizer".format(FLAGS.optimizer))

  config = tf.ConfigProto()
  custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer"
  custom_op.parameter_map["profiling_mode"].b = True

  custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
    '{"output":"./train_url_0/","task_trace":"on"}')
  #custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")
  config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
  config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

  with ms.MonitoredSession(session_creator=ChiefSessionCreator(config=config)) as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()
    for i in range(FLAGS.num_epochs):
      total_time = 0
      total_cost = 0
      i = i+1
      for _ in xrange(i):
        # Training.
        time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                    i)
        total_time += time
        total_cost += cost
      # Results.
      LMFE, MET = print_stats("Epoch {}".format(i), total_cost,
                              total_time, i)
      LMFE = round(LMFE,5)
      costlist.append(LMFE)
      x_list.append(i)
      #print(LMFE, MET)
      #print(costlist)
      plt.plot(x_list,costlist, label='L2L')
      if i%100==0:
        plt.savefig(fname=FLAGS.optimizer,figsize=[100,100])
        plt.show()


if __name__ == "__main__":
  tf.app.run()
