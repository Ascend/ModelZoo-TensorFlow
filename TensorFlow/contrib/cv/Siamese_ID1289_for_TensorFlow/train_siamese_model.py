#! /usr/bin/env python
# -*- coding: utf-8 -*-
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
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Train the model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
import logging
logging.getLogger().setLevel(logging.INFO)
import os
import os.path as osp
import random
import time
from datetime import datetime

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

package_name = 'sacred'
os.system(f'pip install {package_name}')
import pandas as pd
import numpy as np
import tensorflow as tf
from sacred import Experiment
from sacred.observers import FileStorageObserver
import configuration
import siamese_model
from utils.misc_utils import auto_select_gpu, save_cfgs, mkdir_p
from utils.train_utils import load_caffenet

from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
ex = Experiment(configuration.RUN_NAME)
ex.observers.append(FileStorageObserver.create(osp.join(configuration.LOG_DIR, 'sacred')))

@ex.config
def configurations():
  # Add configurations for current script, for more details please see the documentation of `sacred`.
  # REFER: http://sacred.readthedocs.io/en/latest/index.html
  model_config = configuration.MODEL_CONFIG
  train_config = configuration.TRAIN_CONFIG
  track_config = configuration.TRACK_CONFIG

def _configure_learning_rate(train_config, global_step):
  lr_config = train_config['lr_config']

  num_batches_per_epoch = \
    int(train_config['train_data_config']['num_examples_per_epoch'] / train_config['train_data_config']['batch_size'])

  lr_policy = lr_config['policy']
  if lr_policy == 'piecewise_constant':
    lr_boundaries = [int(e * num_batches_per_epoch) for e in lr_config['lr_boundaries']]
    return tf.train.piecewise_constant(global_step,
                                       lr_boundaries,
                                       lr_config['lr_values'])
  elif lr_policy == 'exponential':
    decay_steps = int(num_batches_per_epoch) * lr_config['num_epochs_per_decay']
    return tf.train.exponential_decay(lr_config['initial_lr'],
                                      global_step,
                                      decay_steps=decay_steps,
                                      decay_rate=lr_config['lr_decay_factor'],
                                      staircase=lr_config['staircase'])
  elif lr_policy == 'cosine':
    T_total = train_config['train_data_config']['epoch'] * num_batches_per_epoch
    return 0.5 * lr_config['initial_lr'] * (1 + tf.cos(np.pi * tf.to_float(global_step) / T_total))
  else:
    raise ValueError('Learning rate policy [%s] was not recognized', lr_policy)


def _configure_optimizer(train_config, learning_rate):
  optimizer_config = train_config['optimizer_config']
  optimizer_name = optimizer_config['optimizer'].upper()
  if optimizer_name == 'MOMENTUM':
    optimizer = tf.train.MomentumOptimizer(
      learning_rate,
      momentum=optimizer_config['momentum'],
      use_nesterov=optimizer_config['use_nesterov'],
      name='Momentum')
  elif optimizer_name == 'SGD':
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    refine_optim = tf.train.GradientDescentOptimizer(learning_rate)

    loss_scale_opt = refine_optim
    loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
                                                           decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    optimizer = NPULossScaleOptimizer(loss_scale_opt, loss_scale_manager)
  else:
    raise ValueError('Optimizer [%s] was not recognized', optimizer_config['optimizer'])
  return optimizer

@ex.automain
def main(model_config, train_config, track_config):

  #os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()
  # Create training directory which will be used to save: configurations, model files, TensorBoard logs

  train_dir = train_config['train_dir']
  if not osp.isdir(train_dir):
    logging.info('Creating training directory: %s', train_dir)
    mkdir_p(train_dir)

  g = tf.Graph()
  with g.as_default():
    # Set fixed seed for reproducible experiments
    random.seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    tf.set_random_seed(train_config['seed'])
    # Build the training and validation model
    model = siamese_model.SiameseModel(model_config, train_config, mode='train')
    model.build()

    model_va = siamese_model.SiameseModel(model_config, train_config, mode='validation')
    model_va.build(reuse=True)
    # Save configurations for future reference
    save_cfgs(train_dir, model_config, train_config, track_config)
    learning_rate = _configure_learning_rate(train_config, model.global_step)
    optimizer = _configure_optimizer(train_config, learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)
    logging.info('Trainable variables:')
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
      logging.info('-- {}'.format(v))
    # Set up the training ops
    opt_op = tf.contrib.layers.optimize_loss(
      loss=model.total_loss,
      global_step=model.global_step,
      learning_rate=learning_rate,
      optimizer=optimizer,
      clip_gradients=train_config['clip_gradients'],
      learning_rate_decay_fn=None,
      summaries=['learning_rate'])

    with tf.control_dependencies([opt_op]):
      train_op = tf.no_op(name='train')

    saver = tf.train.Saver(tf.global_variables(),
                           max_to_keep=train_config['max_checkpoints_to_keep'])

    summary_writer = tf.summary.FileWriter(train_dir, g)
    summary_op = tf.summary.merge_all()
    global_variables_init_op = tf.global_variables_initializer()
    local_variables_init_op = tf.local_variables_initializer()

    # Dynamically allocate GPU memory
    #gpu_options = tf.GPUOptions(allow_growth=True)
    #sess_config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.ConfigProto()

    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    #custom_op.parameter_map["dynamic_input"].b = 1
    #custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes('lazy_recompile')
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    #config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    #config = npu_tf_config.session_dump_config(config, action='fusion_off')
    #custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/ma-user/modelarts/inputs/data_url_0/Logs")
    #custom_op.parameter_map["enable_dump_debug"].b = True
    #custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
    sess = tf.Session(config=config)
    model_path = tf.train.latest_checkpoint(train_config['train_dir'])
    if not model_path:
      sess.run(global_variables_init_op)
      sess.run(local_variables_init_op)
      start_step = 0
      if model_config['sa_siam_config']['en_semantic']:
        load_caffenet(train_config['caffenet_dir'], sess)
      if model_config['embed_config']['embedding_checkpoint_file']:
        model.init_fn(sess)

    else:
      logging.info('Restore from last checkpoint: {}'.format(model_path))
      sess.run(local_variables_init_op)
      saver.restore(sess, model_path)
      start_step = tf.train.global_step(sess, model.global_step.name) + 1

    g.finalize()  # Finalize graph to avoid adding ops by mistake
    # Training loop
    data_config = train_config['train_data_config']
    total_steps = int(data_config['epoch'] *
                      data_config['num_examples_per_epoch'] /
                      data_config['batch_size'])
    #total_steps = 1000
    logging.info('Train for {} steps'.format(total_steps))

    #i = 0
    #b = np.zeros(2000)
    df = pd.DataFrame(columns=['Step', 'Loss', 'batch_loss', ])  # 列名

    df.to_csv(osp.join(train_config['output_dir'], 'appearance_npu.csv'), index=False)  # 路径可以根据需要更改
    for step in range(start_step, total_steps):
      start_time = time.time()
      _, loss, batch_loss = sess.run([train_op, model.total_loss, model.batch_loss])

      duration = time.time() - start_time
      if step % 10 == 0:
        examples_per_sec = data_config['batch_size'] / float(duration)
        time_remain = data_config['batch_size'] * (total_steps - step) / examples_per_sec
        m, s = divmod(time_remain, 60)
        h, m = divmod(m, 60)
        format_str = ('%s: step %d, total loss = %.2f batch loss = %.2f ( %.1f examples/sec; %.3f '
                      'sec/batch; %dh:%02dm:%02ds remains)')
        logging.info(format_str % (datetime.now(), step, loss, batch_loss,
                                   examples_per_sec, duration, h, m, s))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
        #b[i] = loss
        #i = i+1
        Step = "step %d" % step
        Loss = "%f" % loss
        Batch_loss = "%f" % batch_loss
        # 将数据保存在一维列表
        list = [Step, Loss, Batch_loss]
        # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
        data = pd.DataFrame([list])
        data.to_csv(osp.join(train_config['output_dir'], 'appearance_npu.csv'), mode='a', header=False,
                  index=False)  # mode设为a,就可以向csv文件追加数据了
      if (step + 1) % train_config['save_model_every_n_step'] == 0 or (step + 1) == total_steps:
        checkpoint_path = osp.join(train_config['train_dir'], 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


