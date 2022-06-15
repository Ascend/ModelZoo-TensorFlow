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

"""Script to train the Attention OCR model.

A simple usage example:
python train.py
"""
# from npu_bridge.npu_init import *
import collections
import logging
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.compat.v1 import flags
from tensorflow.contrib.tfprof import model_analyzer
from tensorflow.python.training import checkpoint_management

import data_provider
import common_flags

from eval import eval_once
from my_slim_learning import train as slim_train
import os
import time
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from npu_bridge.npu_init import *

FLAGS = flags.FLAGS
common_flags.define()

# yapf: disable
flags.DEFINE_integer('task', 0,
                     'The Task ID. This value is used when training with '
                     'multiple workers to identify each worker.')

flags.DEFINE_integer('ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then'
                     ' the parameters are handled locally by the worker.')

flags.DEFINE_integer('save_summaries_secs', 60,
                     'The frequency with which summaries are saved, in '
                     'seconds.')

flags.DEFINE_integer('save_interval_secs', 600,
                     'Frequency in seconds of saving the model.')

flags.DEFINE_integer('max_number_of_steps', int(1e6),
                     'The maximum number of gradient steps.')

flags.DEFINE_integer('log_interval_steps', 25000,
                     'The log interval number of steps.')

# '/home/yuqiao/TF_model/research/attention_ocr/python/inception_v3.ckpt'
# flags.DEFINE_string('checkpoint_inception', './inception_v3.ckpt',
#                     'Checkpoint to recover inception weights from.')
flags.DEFINE_string('checkpoint_inception', '/home/yuqiao/TF_model/research/attention_ocr/python/inception_v3.ckpt',
                    'Checkpoint to recover inception weights from.')

flags.DEFINE_float('clip_gradient_norm', 2.0,
                   'If greater than 0 then the gradients would be clipped by '
                   'it.')

flags.DEFINE_bool('sync_replicas', False,
                  'If True will synchronize replicas during training.')

flags.DEFINE_integer('replicas_to_aggregate', 1,
                     'The number of gradients updates before updating params.')

flags.DEFINE_integer('total_num_replicas', 1,
                     'Total number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_boolean('reset_train_dir', False,
                     'If true will delete all files in the train_log_dir')

flags.DEFINE_boolean('show_graph_stats', False,
                     'Output model size stats to stderr.')

# flags.DEFINE_string('eval_log_dir', '/tmp/attention_ocr/eval',
#                     'Directory where the evaluation results are saved to.')
# yapf: enable

TrainingHParams = collections.namedtuple('TrainingHParams', [
    'learning_rate',
    'optimizer',
    'momentum',
    'use_augment_input',
])


def get_training_hparams():
    return TrainingHParams(
        learning_rate=FLAGS.learning_rate,
        optimizer=FLAGS.optimizer,
        momentum=FLAGS.momentum,
        use_augment_input=FLAGS.use_augment_input)


def create_optimizer(hparams):
    """Creates optimized based on the specified flags."""
    if hparams.optimizer == 'momentum':
        optimizer = tf.compat.v1.train.MomentumOptimizer(
            hparams.learning_rate, momentum=hparams.momentum)
    elif hparams.optimizer == 'adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(hparams.learning_rate)
    elif hparams.optimizer == 'adadelta':
        optimizer = tf.compat.v1.train.AdadeltaOptimizer(hparams.learning_rate)
    elif hparams.optimizer == 'adagrad':
        optimizer = tf.compat.v1.train.AdagradOptimizer(hparams.learning_rate)
    elif hparams.optimizer == 'rmsprop':
        optimizer = tf.compat.v1.train.RMSPropOptimizer(
            hparams.learning_rate, momentum=hparams.momentum)
    return optimizer



def train(loss, init_fn, hparams, iterator, eval_ops_test=None):
    """Wraps slim.learning.train to run a training loop.

    Args:
      loss: a loss tensor
      init_fn: A callable to be executed after all other initialization is done.
      hparams: a model hyper parameters
    """
    #eval

    #end eval




    optimizer = create_optimizer(hparams)

    if FLAGS.sync_replicas:
        replica_id = tf.constant(FLAGS.task, tf.int32, shape=())
        optimizer = tf.LegacySyncReplicasOptimizer(
            opt=optimizer,
            replicas_to_aggregate=FLAGS.replicas_to_aggregate,
            replica_id=replica_id,
            total_num_replicas=FLAGS.total_num_replicas)
        sync_optimizer = optimizer
        startup_delay_steps = 0
    else:
        startup_delay_steps = 0
        sync_optimizer = None

    train_op = slim.learning.create_train_op(
        loss,
        optimizer,
        summarize_gradients=True,
        clip_gradient_norm=FLAGS.clip_gradient_norm)



    # config = tf.ConfigProto()
    # custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    # custom_op.name = "NpuOptimizer"
    # config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    # config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

    saver = tf.train.Saver(max_to_keep=5)

    # with tf.Session(config=config) as session:
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        iterator.initializer.run()

        number_of_steps = FLAGS.max_number_of_steps
        log_interval_steps = FLAGS.log_interval_steps
        logdir = FLAGS.train_log_dir

        step_index = 0
        init_fn(session)
        checkpoint_dir = logdir
        ckpt = checkpoint_management.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            step_index = int(ckpt.model_checkpoint_path.split("-")[-1])
            saver.restore(session, ckpt.model_checkpoint_path)
            # saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
            # if eval_ops_test:
            #     eva = session.run(eval_ops_test)
            #     print("eval:{}".format(eva))
            # eval_once()
        while True:
            if number_of_steps is not None and step_index >= number_of_steps:
                break
            step_index += 1
            step_start = time.time()
            total_loss = session.run(train_op)
            step_end = time.time()
            print("global step {:d}: loss = {:.4f} ({:.3f} sec/step)".format(step_index, total_loss, step_end - step_start))
            if step_index % log_interval_steps == 0:
                saver.save(session, logdir + os.sep + 'model.ckpt', global_step=step_index)
                # eval_once()

def prepare_training_dir():
    if not tf.io.gfile.exists(FLAGS.train_log_dir):
        logging.info('Create a new training directory %s', FLAGS.train_log_dir)
        tf.io.gfile.makedirs(FLAGS.train_log_dir)
    else:
        if FLAGS.reset_train_dir:
            logging.info('Reset the training directory %s', FLAGS.train_log_dir)
            tf.io.gfile.rmtree(FLAGS.train_log_dir)
            tf.io.gfile.makedirs(FLAGS.train_log_dir)
        else:
            logging.info('Use already existing training directory %s',
                         FLAGS.train_log_dir)


def calculate_graph_metrics():
    param_stats = model_analyzer.print_model_analysis(
        tf.compat.v1.get_default_graph(),
        tfprof_options=model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    return param_stats.total_parameters


def main(_):
    prepare_training_dir()

    dataset = common_flags.create_dataset(split_name=FLAGS.split_name)

    model = common_flags.create_model(dataset.num_char_classes,
                                      dataset.max_sequence_length,
                                      dataset.num_of_views, dataset.null_code)
    hparams = get_training_hparams()

    device_setter = tf.compat.v1.train.replica_device_setter(
        FLAGS.ps_tasks, merge_devices=True)
    # with tf.device('/cpu:0'):
    with tf.device(device_setter):
        data, iterator = data_provider.my_get_data(
            dataset,
            FLAGS.batch_size,
            augment=hparams.use_augment_input,
            central_crop_size=common_flags.get_crop_size())
        endpoints = model.create_base(data.images, data.labels_one_hot)
        # new_eval()
        total_loss = model.create_loss(data, endpoints)
        model.create_summaries(data, endpoints, dataset.charset, is_training=True)
        eval_ops_test = model.create_summaries(data, endpoints, dataset.charset, is_training=False)
        FLAGS.checkpoint_inception='/home/yuqiao/TF_model/research/attention_ocr/python/inception_v3.ckpt'
        init_fn = model.create_init_fn_to_restore(FLAGS.checkpoint,
                                                  FLAGS.checkpoint_inception)
        if FLAGS.show_graph_stats:
            logging.info('Total number of weights in the graph: %s',
                         calculate_graph_metrics())
        train(total_loss, init_fn, hparams, iterator, eval_ops_test)


if __name__ == '__main__':
    app.run()

