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
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import random
import time
from datetime import datetime

# import npu_bridge
from npu_bridge.npu_init import *

import numpy as np
import tensorflow as tf

from models.bisenet import BiseNet

import configuration
from utils.misc_utils import mkdir_p, save_cfgs

import argparse


def _configure_learning_rate(train_config, global_step):
    lr_config = train_config['lr_config']

    num_batches_per_epoch = \
        int(train_config['train_data_config']['num_examples_per_epoch']
            / train_config['train_data_config']['batch_size'])

    lr_policy = lr_config['policy']
    if lr_policy == 'piecewise_constant':
        lr_boundaries = [int(e * num_batches_per_epoch)
                         for e in lr_config['lr_boundaries']]
        return tf.train.piecewise_constant(global_step,
                                           lr_boundaries,
                                           lr_config['lr_values'])
    elif lr_policy == 'exponential':
        decay_steps = int(num_batches_per_epoch)  \
            * lr_config['num_epochs_per_decay']
        return tf.train.exponential_decay(lr_config['initial_lr'],
                                          global_step,
                                          decay_steps=decay_steps,
                                          decay_rate=lr_config['lr_decay_factor'],
                                          staircase=lr_config['staircase'])
    elif lr_policy == 'polynomial':
        T_total = (int(num_batches_per_epoch)+1)  \
            * train_config['train_data_config']['epoch']
        return lr_config['initial_lr'] * (1 - tf.cast(global_step, dtype=tf.float32)/T_total)**lr_config['power']
    elif lr_policy == 'cosine':
        T_total = train_config['train_data_config']['epoch']  \
            * num_batches_per_epoch
        return 0.5 * lr_config['initial_lr'] * (1 + tf.cos(np.pi * tf.cast(global_step, dtype=tf.float32) / T_total))
    else:
        raise ValueError(
            'Learning rate policy [%s] was not recognized', lr_policy)


def _configure_optimizer(train_config, learning_rate):
    optimizer_config = train_config['optimizer_config']
    optimizer_name = optimizer_config['optimizer'].upper()
    if optimizer_name == 'MOMENTUM':
        optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate,
            momentum=optimizer_config['momentum'],
            use_nesterov=optimizer_config['use_nesterov'],
            name='Momentum')
    elif optimizer_name == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer_name == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate, optimizer_config['decay'], optimizer_config['momentum'])
    else:
        raise ValueError(
            'Optimizer [%s] was not recognized', optimizer_config['optimizer'])
    return optimizer


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="CamVid")
    parser.add_argument("--class_dict", type=str, default="./CamVid/class_dict.csv")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_epochs", type=int, default=2)
    parser.add_argument("--load_checkpoint", type=bool, default=False)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()

    model_config = configuration.MODEL_CONFIG
    train_config = configuration.TRAIN_CONFIG

    train_config['DataSet'] = args.data_path
    train_config['class_dict'] = args.class_dict
    train_config['train_data_config']['batch_size'] = args.batch_size
    train_config['validation_data_config']['batch_size'] = args.batch_size
    train_config['test_data_config']['batch_size'] = args.batch_size
    train_config['train_data_config']['epoch'] = args.train_epochs

    num_classes = 32    # TODO: num_classes need to fix

    train_dir = train_config['train_dir']
    if not os.path.isdir(train_dir):
        logging.info('Creating training directory: %s', train_dir)
        mkdir_p(train_dir)

    g = tf.Graph()
    with g.as_default():
        # Set fixed seed for reproducible experiments
        random.seed(train_config['seed'])
        np.random.seed(train_config['seed'])
        tf.compat.v1.set_random_seed(train_config['seed'])

        # Build the training and validation model
        model = BiseNet(model_config, train_config, num_classes, mode="train")
        model.build()
        model_va = BiseNet(model_config, train_config,
                           num_classes, mode="validation")
        model_va.build(reuse=True)

        # Save configurations for future reference
        save_cfgs(train_dir, model_config, train_config)

        learning_rate = _configure_learning_rate(
            train_config, model.global_step)
        optimizer = _configure_optimizer(train_config, learning_rate)
        tf.compat.v1.summary.scalar('learning_rate', learning_rate)
        update_ops = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS)
            
        opt_tmp = optimizer
        loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        optimizer = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)
        with tf.control_dependencies(update_ops):
            train_op = tf.contrib.layers.optimize_loss(loss=model.total_loss,
                                                       global_step=model.global_step,
                                                       learning_rate=learning_rate,
                                                       optimizer=optimizer,
                                                       clip_gradients=train_config['clip_gradients'],
                                                       learning_rate_decay_fn=None,
                                                       summaries=['learning_rate'])

        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(),
                                         max_to_keep=train_config['max_checkpoints_to_keep'])

        global_variables_init_op = tf.compat.v1.global_variables_initializer()
        local_variables_init_op = tf.compat.v1.local_variables_initializer()
        g.finalize()  # Finalize graph to avoid adding ops by mistake
        
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"

        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
        
        sess = tf.compat.v1.Session(config=config)
        model_path = tf.train.latest_checkpoint(train_config['train_dir'])
        
        if args.load_checkpoint:
            if not model_path:
                raise FileNotFoundError
            logging.info('Restore from last checkpoint: {}'.format(model_path))
            sess.run(local_variables_init_op)
            saver.restore(sess, model_path)
            start_step = tf.compat.v1.train.global_step(
                sess, model.global_step.name) + 1
        else:
            sess.run(global_variables_init_op)
            sess.run(local_variables_init_op)
            start_step = 0

            if model_config['frontend_config']['pretrained_dir'] and model.init_fn:
                model.init_fn(sess)

        data_config = train_config['train_data_config']
        
        total_steps = int(data_config['epoch'] 
                        * data_config['num_examples_per_epoch'] 
                        / data_config['batch_size'])
        
        logging.info('Train for {} steps'.format(total_steps))

        begin_time = time.time()

        for step in range(start_step, total_steps):
            start_time = time.time()
            _, predict_loss, loss, accuracy, mean_IOU = sess.run(
                [train_op, model.loss, model.total_loss, model.accuracy, model.mean_IOU])
            duration = time.time() - start_time

            # 打印日志
            if step % 10 == 0:
                FPS = data_config['batch_size'] / float(duration)
                time_remain = data_config['batch_size'] * (total_steps - step) / FPS
                m, s = divmod(time_remain, 60)
                h, m = divmod(m, 60)
                format_str = ('%s: step %d '
                              'loss = %.2f ' # predict loss = %.2f
                              'accuracy = %.2f mean IOU = %.2f '
                              'FPS = %.1f step time = %.2f ' 
                              '(%dh:%02dm:%02ds remains)')
                logging.info(format_str % (datetime.now(), step, loss, # predict_loss,
                                            accuracy[0], mean_IOU[0],
                                            FPS, duration,
                                            h, m, s))

            if step % train_config['save_model_every_n_step'] == 0 or (step + 1) == total_steps:
                checkpoint_path = os.path.join(
                    train_config['train_dir'], 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        logging.info('%s: Train finish after %d steps' % (datetime.now(), total_steps))
        sess.close()

if __name__ == "__main__":
    main()
