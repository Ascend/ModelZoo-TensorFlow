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

r"""Training code based on TF Eager."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import npu_device

print('npu_device loaded')

# npu_device.global_options().precision_mode = 'allow_fp32_to_fp16'
# npu_config = {}
# npu_device.open().as_default()



import os

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf

from algorithms import get_algo
from config import CONFIG
from datasets import create_dataset
from utils import get_lr_fn
from utils import get_lr_opt_global_step
from utils import restore_ckpt
from utils import setup_train_dir
from utils import Stopwatch


#import npu_device

#print('npu_device loaded')

#npu_device.global_options().precision_mode = 'allow_fp32_to_fp16'
#npu_config = {}
#npu_device.open().as_default()


# flags.DEFINE_string('logdir', '/home/zWX1053564/code/ID0714_TCC/tcc/tmp/alignment_logs', 'Path to logs.')
flags.DEFINE_string('logdir', 'tmp/alignment_logs', 'Path to logs.')
flags.DEFINE_boolean('defun', True, 'Defun functions in algo for faster '
                                    'training.')
flags.DEFINE_boolean('debug', False, 'Plots detailed summaries on Tensorboard.')
flags.DEFINE_boolean(
    'force_train', False, 'Continue with training even when '
                          'train_logs exist. Useful if one has to resume training. '
                          'By default switched off to prevent overwriting existing '
                          'experiments.')
flags.DEFINE_boolean('visualize', False, 'Visualize images, gradients etc. '
                                         'Switched off by for default to speed training up and '
                                         'takes less memory.')
flags.DEFINE_string(name='precision_mode', default= 'allow_mix_precision',
                    help='allow_fp32_to_fp16/force_fp16/ ' 
                    'must_keep_origin_dtype/allow_mix_precision.')
flags.DEFINE_boolean(name='over_dump', default=False,
                    help='if or not over detection, default is False')
flags.DEFINE_boolean(name='data_dump_flag', default=False,
                    help='data dump flag, default is False')
flags.DEFINE_string(name='data_dump_step', default="10",
                    help='data dump step, default is 10')
flags.DEFINE_boolean(name='profiling', default=False,
                    help='if or not profiling for performance debug, default is False') 
flags.DEFINE_string(name='profiling_dump_path', default="/home/data",
                    help='the path to save profiling data')                                      
flags.DEFINE_string(name='over_dump_path', default="/home/data",
                    help='the path to save over dump data')  
flags.DEFINE_string(name='data_dump_path', default="/home/data",
                    help='the path to save dump data')     
flags.DEFINE_boolean(name='use_mixlist', default=True,
                    help='whether to enable mixlist, default is True')
flags.DEFINE_string(name='mixlist_file', default='ops_info.json',
                    help='mixlist file name, default is ops_info.json')
FLAGS = flags.FLAGS
layers = tf.keras.layers

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# f_loss = open('./loss_train.txt','w+',encoding='utf-8')

def npu_config():
  FLAGS = flags.FLAGS
  npu_config = {}

  if FLAGS.data_dump_flag:
    npu_device.global_options().dump_config.enable_dump = True
    npu_device.global_options().dump_config.dump_path = FLAGS.data_dump_path
    npu_device.global_options().dump_config.dump_step = FLAGS.data_dump_step
    npu_device.global_options().dump_config.dump_mode = "all"

  if FLAGS.over_dump:
    npu_device.global_options().dump_config.enable_dump_debug = True
    npu_device.global_options().dump_config.dump_path = FLAGS.over_dump_path
    npu_device.global_options().dump_config.dump_debug_mode = "all"

  if FLAGS.profiling:
    npu_device.global_options().profiling_config.enable_profiling = True
    profiling_options = '{"output":"' + FLAGS.profiling_dump_path + '", \
                        "training_trace":"on", \
                        "task_trace":"on", \
                        "aicpu":"on", \
                        "aic_metrics":"PipeUtilization",\
                        "fp_point":"model/transformer_v2/Transformer/attention_bias/mul", \
                        "bp_point":"gradient_tape/model/transformer_v2/Transformer/encode/embedding_shared_weights/embedding/mul/Mul"}'
    npu_device.global_options().profiling_config.profiling_options = profiling_options
  npu_device.global_options().precision_mode=FLAGS.precision_mode
  #npu_device.global_options().fusion_switch_file='../configs/fusion_off.cfg'
  if FLAGS.use_mixlist and FLAGS.precision_mode=='allow_mix_precision':
    logging.info('start to set op blacklist according to %s',FLAGS.mixlist_file)
    npu_device.global_options().modify_mixlist="../configs/"+FLAGS.mixlist_file
  npu_device.open().as_default()

def train():
    """Trains model and evaluates on relevant downstream tasks."""
    CONFIG.LOGDIR = FLAGS.logdir
    logdir = CONFIG.LOGDIR
    setup_train_dir(logdir)

    # Common code for multigpu and single gpu. Set devices here if you don't
    # want to use all the GPUs on the machine. Default is to use all GPUs.
#    strategy = tf.distribute.MirroredStrate(devices=["/gpu:0"])
    strategy = tf.distribute.OneDeviceStrategy(device='/cpu:0')
    # strategy = tf.distribute.MirroredStrategy(cross_device_ops='hccl')
    # strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        algo = get_algo(CONFIG.TRAINING_ALGO)
        # print('------algo-------\n')
        # print(algo)
        # print('-----------------\n')
        # Setup summary writer.
        summary_writer = tf.summary.create_file_writer(
            os.path.join(logdir, 'train_logs'), flush_millis=10000)

        learning_rate, optimizer, global_step = get_lr_opt_global_step()
        ckpt_manager, _, _ = restore_ckpt(
            logdir=logdir, optimizer=optimizer, **algo.model)

        global_step_value = global_step.numpy()

        # Remember in Eager mode learning rate variable needs to be updated
        # manually. Calling lr_fn each iteration to get current learning rate.
        lr_fn = get_lr_fn(CONFIG.OPTIMIZER)

        # Setup Dataset Iterators from train and val datasets.
        batch_size_per_replica = CONFIG.TRAIN.BATCH_SIZE

        total_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
        train_ds = create_dataset('train', mode='train',
                                  batch_size=total_batch_size,
                                  return_iterator=False)
        #train_iterator = strategy.make_dataset_iterator(train_ds)
        train_iterator = iter(train_ds)
        # train_iterator = strategy.experimental_distribute_dataset(train_ds)

        def train_step(data):
            steps = data['chosen_steps']
            seq_lens = data['seq_lens']
            loss = algo.train_one_iter(data, steps, seq_lens, global_step, optimizer)
            return loss

        # This reduction only affects reporting, not the gradients.
        # pylint: disable=g-long-lambda
        dist_train = lambda it: strategy.reduce(
            tf.distribute.ReduceOp.SUM, strategy.experimental_run(train_step, it),
            axis=None)
        #             tf.distribute.ReduceOp.SUM, strategy.experimental_run(train_step, it),
        # pylint: enable=g-long-lambda
        if FLAGS.defun:
            dist_train = tf.function(dist_train)

        stopwatch = Stopwatch()

        try:
            while global_step_value < CONFIG.TRAIN.MAX_ITERS:
                with summary_writer.as_default():
                    with tf.summary.record_if(
                            global_step_value % CONFIG.LOGGING.REPORT_INTERVAL == 0):
                        loss = dist_train(train_iterator)
                        loss_v = loss.cpu().numpy()
                        #print(loss_v,type(loss_v))
                        # f_loss.write('{:.3f}'.format(loss_v)+'\n')
                        #print(loss_v,type(loss_v))
                        # Update learning rate based in lr_fn.
                        learning_rate.assign(lr_fn(learning_rate, global_step))

                        tf.summary.scalar('loss', loss, step=global_step)
                        tf.summary.scalar('learning_rate', learning_rate, step=global_step)

                        # Save checkpoint.
                        if global_step_value % CONFIG.CHECKPOINT.SAVE_INTERVAL == 0:
                            ckpt_manager.save()
                            logging.info('Checkpoint saved at iter %d.', global_step_value)

                        # Update global step.
                        global_step_value = global_step.numpy()

                        time_per_iter = stopwatch.elapsed()

                        tf.summary.scalar(
                            'timing/time_per_iter', time_per_iter, step=global_step)

                        logging.info('Iter[{}/{}], {:.1f}s/iter, Loss: {:.3f}'.format(
                            global_step_value, CONFIG.TRAIN.MAX_ITERS, time_per_iter,
                            loss.numpy()))
                        #f_loss.write(str(loss.numpy())+'\n')
                        # Reset stopwatch after iter is complete.
                        stopwatch.reset()

        except KeyboardInterrupt:
            logging.info('Caught keyboard interrupt. Saving model before quitting.')

        finally:
            # Save the final checkpoint.
            ckpt_manager.save()
            logging.info('Checkpoint saved at iter %d', global_step_value)
        logging.info('finally OKKKKKKKK')


def main(_):
    npu_config()
    tf.enable_v2_behavior()
    tf.keras.backend.set_learning_phase(1) # 学习阶段标志是一个布尔张量（0 = test，1 = train），它作为输入传递给任何 Keras 函数，以在训练和测试时执行不同的行为操作
    train()


if __name__ == '__main__':
    app.run(main)
    # main()
