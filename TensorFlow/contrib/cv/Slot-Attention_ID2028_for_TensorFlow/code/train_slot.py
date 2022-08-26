# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Training loop for object discovery with Slot Attention."""
from npu_bridge.npu_init import *
import datetime
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
# import moxing as mox
# from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
# from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
# from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager

import os
# os.system("pip3 install tensorflow_datasets==3.0.0")
import data as data_utils
import model as model_utils
import utils as utils


FLAGS = flags.FLAGS
flags.DEFINE_string('data_url', '', 'dataurl')
flags.DEFINE_string('train_url', '', 'trainurl')
flags.DEFINE_string("model_dir", "/mnt/home/test_user01/pr_slot_attention/checkp",
                    "Where to save the checkpoints.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 64, "Batch size for the model.")
flags.DEFINE_integer("num_slots", 7, "Number of slots in Slot Attention.")
flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")
flags.DEFINE_float("learning_rate", 0.0004, "Learning rate.")
flags.DEFINE_integer("num_train_steps", 500000, "Number of training steps.")
flags.DEFINE_integer("warmup_steps", 10000,
                     "Number of warmup steps for the learning rate.")
flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
flags.DEFINE_integer("decay_steps", 100000,
                     "Number of steps for the learning rate decay.")


def main(argv):
  del argv
  # Hyperparameters of the model.
  batch_size = FLAGS.batch_size
  num_slots = FLAGS.num_slots
  num_iterations = FLAGS.num_iterations
  base_learning_rate = FLAGS.learning_rate
  num_train_steps = FLAGS.num_train_steps
  warmup_steps = FLAGS.warmup_steps
  decay_rate = FLAGS.decay_rate
  decay_steps = FLAGS.decay_steps
  tf.random.set_random_seed(FLAGS.seed)
  #tf.random.set_seed(FLAGS.seed)
  resolution = (128, 128)
  # Build dataset iterators, optimizers and model.
  data_iterator = data_utils.build_clevr_iterator(
      batch_size, split="train", resolution=resolution, shuffle=True,
      max_n_objects=6, get_properties=False, apply_crop=True)

  model = model_utils.build_model(resolution, batch_size, num_slots,
                                  num_iterations, model_type="object_discovery")
  global_steps = tf.Variable(
    0, trainable=False, name="global_step", dtype=tf.int64)

  X = tf.placeholder(dtype=tf.float32, shape=[64, 128, 128, 3])
  lrflag = tf.placeholder(tf.int16)
  preds = model(X, training=True)
  recon_combined, recons, masks, slots = preds
  loss_value = tf.reduce_mean(tf.math.squared_difference(recon_combined, X))
  lr = tf.Variable(0.0004, dtype=tf.float32)
  # trainstep=tf.train.AdamOptimizer(lr).minimize(loss_value,global_step=global_steps)

  #添加loss_scale
  opt=tf.train.AdamOptimizer(lr)
  loss_scale_manager=ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
                                                           decr_every_n_nan_or_inf=2,decr_ratio=0.5)
  loss_scale_optimizer=NPULossScaleOptimizer(opt, loss_scale_manager)
  trainstep=loss_scale_optimizer.minimize(loss_value,global_step=global_steps)

  # new_value = tf.add(global_steps, 1)
  # update=tf.assign(global_steps,new_value)
  el = data_iterator.get_next()
  compa = tf.less(global_steps, warmup_steps)


  lrresult = tf.cond(lrflag>0, lambda: base_learning_rate * tf.cast(
          global_steps, tf.float32) / tf.cast(warmup_steps, tf.float32), lambda: base_learning_rate)
  learning_rate = lrresult * (decay_rate ** (
          tf.cast(global_steps, tf.float32) / tf.cast(decay_steps, tf.float32)))
  lr=tf.assign(lr, learning_rate)

  init = tf.global_variables_initializer()
  # Prepare checkpoint manager.
  saver = tf.train.Saver(max_to_keep=5)

  save_path=tf.train.latest_checkpoint(FLAGS.model_dir)
  tf.disable_eager_execution()
  # profiling_dir = "/cache/profiling"
  # os.makedirs(profiling_dir)
  config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  config.gpu_options.allow_growth = True
  custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
  custom_op.name = "NpuOptimizer"
  custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes("allow_mix_precision")
  config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
  config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
  #添加自定义算子
  custom_op.parameter_map["customize_dtypes"].s=tf.compat.as_bytes("./code/switch_config.txt")
  # custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("./ops_info.json")
  # custom_op.parameter_map["profiling_mode"].b = True
  # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/cache/profiling","task_trace":"on"}')
  sess = tf.Session(config=config)
  if save_path:
    saver.restore(sess, save_path=save_path)
    logging.info("Restored from %s", save_path)
  else:
    logging.info("Initializing from scratch.")

  sess.run(init)

  for _ in range(num_train_steps):
    start = time.time()
    batch = sess.run(el)
    # Learning rate warm-up.
    flag1=sess.run(compa)
    if flag1:
      finallr=sess.run(lr,feed_dict={lrflag:1})
    else:
      finallr=sess.run(lr, feed_dict={lrflag:-1})
    loss,_=sess.run([loss_value,trainstep],feed_dict={X:batch["image"]})

    # Update the global step. We update it before logging the loss and saving
    # the model so that the last checkpoint is saved at the last iteration.
    #sess.run(update)
    gs=sess.run(global_steps)
    # Log the training loss.
    if not gs % 1:
      logging.info("Step: %s, Loss: %.6f, Time: %s",
                   gs, loss, time.time() - start)
      # mox.file.copy_parallel("/cache/profiling", "obs://lwr-slot-npu/profiling")

    # We save the checkpoints every 1000 iterations.
    if not gs  % 1000:
      # Save the checkpoint of the model.
      path = '{}/checkpoint.ckpt'.format(FLAGS.model_dir)
      saved_ckpt=saver.save(sess,path, global_step=gs)
      logging.info("Saved checkpoint: %s", saved_ckpt)
      graphpath = '/mnt/home/test_user01/pr_slot_attention/graphcheckp'
      tf.io.write_graph(sess.graph, graphpath, 'graph.pbtxt', as_text=True)
      logging.info('Save graph {} at iteration {}'.format(graphpath, gs))


if __name__ == "__main__":
  app.run(main)

