"""
train
"""
# coding=utf-8
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
from npu_bridge.npu_init import NPULossScaleOptimizer, npu_config_proto, RewriterConfig, \
    ExponentialUpdateLossScaleManager, FixedLossScaleManager
# import precision_tool.tf_config as npu_tf_config
import os

# os.system('pip3 install ml_collections')
# os.system('pip3 install tensorflow_datasets==3.0.0')
# os.system('pip3 install pyopenssl')
import time
from datetime import datetime

# from absl import app
# from absl import flags
# from absl import logging

from ml_collections import config_flags
import tensorflow as tf
import tensorflow_datasets as tfds
# import moxing as mx
import datasets
from models import colorizer
from models import upsampler
from utils import train_utils

flags = tf.app.flags
# FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', [
    'train', 'eval_train', 'eval_valid', 'eval_test'], 'Operation mode.')
flags.DEFINE_string(
    'train_url', 'obs://imagenet2012-lp/coltran_log/', 'the path of train log in obs')
flags.DEFINE_string('data_url', 'obs://imagenet2012-lp/coltran_re/imagenet2012/',
                    'the path of train data in obs')
flags.DEFINE_string('logdir', '/cache/saveModels/logfile_coltran_spatial_upsampler', 'Main directory for logs.')
flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
flags.DEFINE_enum('accelerator_type', 'GPU', ['CPU', 'GPU'],
                  'Hardware type.')
flags.DEFINE_enum('dataset', 'imagenet', ['imagenet', 'custom'], 'Dataset')
flags.DEFINE_string('data_dir', '/home/ma-user/modelarts/inputs/data_url_0/', 'Data directory for custom images.')
# flags.DEFINE_string('tpu_worker_name', 'tpu_worker', 'Name of the TPU worker.')
flags.DEFINE_bool(
    'restore', False, 'Finetune from a pretrained checkpoint.')
flags.DEFINE_string(
    'pretrain_dir',
    'obs://imagenet2012-lp/coltran_log/MA-new-coltran_modelarts-05-12-17-36/output/logfile_coltran_colorizer/',
    'Finetune from a pretrained checkpoint.')
flags.DEFINE_string('prelogdir', '/cache/saveModels/logfile_coltran_colorizer', 'pre train logs.')
flags.DEFINE_string('summaries_log_dir', 'summaries', 'Summaries parent.')
flags.DEFINE_integer('steps_per_summaries', 100, 'Steps per summaries.')
flags.DEFINE_integer('devices_per_worker', 1, 'Number of devices per worker.')
flags.DEFINE_integer('num_workers', 1, 'Number workers.')
config_flags.DEFINE_config_file(
    'config',
    default='./configs/spatial_upsampler.py',
    help_string='Training configuration file.')
FLAGS = flags.FLAGS
if not os.path.exists(FLAGS.logdir):
    os.makedirs(FLAGS.logdir)
if FLAGS.restore:
    os.makedirs(FLAGS.prelogdir)
    # mx.file.copy_parallel(FLAGS.pretrain_dir, FLAGS.prelogdir)


def loss_on_batch(inputs, model, config, training=False):
    """
    Loss Function.
    Loss on a batch of inputs.
    Args:
        inputs: A dictionary of input tensors keyed by name.
        model: A model function.
        config: A dictionary of configuration parameters.
        training: Whether this is a training run.
    """
    logits, aux_output = model.get_logits(
        inputs_dict=inputs, train_config=config, training=training)
    loss, aux_loss_dict = model.loss(
        targets=inputs, logits=logits, train_config=config, training=training,
        aux_output=aux_output)
    loss_factor = config.get('loss_factor', 1.0)

    total_loss = loss_factor * loss

    for aux_key, aux_loss in aux_loss_dict.items():
        aux_loss_factor = config.get(f'{aux_key}_loss_factor', 1.0)
        total_loss += aux_loss_factor * aux_loss
    return total_loss


def train():
    """
    Train the model.
    """
    config = FLAGS.config
    tf.reset_default_graph()
    with tf.Graph().as_default():
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        def input_fn(input_context=None):
            read_config = None
            if input_context is not None:
                read_config = tfds.ReadConfig(input_context=input_context)

            dataset = datasets.get_dataset(
                name=FLAGS.dataset,
                config=config,
                batch_size=config.batch_size,
                subset='train',
                read_config=read_config,
                data_dir=FLAGS.data_dir)
            return dataset

        # DATASET CREATION.
        batch_size = config.batch_size
        train_dataset = train_utils.dataset_with_strategy(input_fn, None)
        iterator = tf.data.make_initializable_iterator(train_dataset)
        data_iterator = iterator.get_next('getnext')
        ## build model
        # downsample = config.get('downsample', False)
        # downsample_res = config.get('downsample_res', 64)
        # h, w = config.resolution
        if config.model.name == 'coltran_core':
            # if downsample:
            # h, w = downsample_res, downsample_res
            # zero = tf.zeros((batch_size, h, w, 3), dtype=tf.int32)
            model = colorizer.ColTranCore(config.model)
            # model(zero, training=True)

        # c = 1 if True else 3
        if config.model.name == 'color_upsampler':
            # if downsample:
            # h, w = downsample_res, downsample_res
            # zero_slice = tf.zeros((batch_size, h, w, c), dtype=tf.int32)
            # zero = tf.zeros((batch_size, h, w, 3), dtype=tf.int32)
            model = upsampler.ColorUpsampler(config.model)
            # model(zero, inputs_slice=zero_slice, training=True)
        elif config.model.name == 'spatial_upsampler':
            # zero_slice = tf.zeros((batch_size, h, w, c), dtype=tf.int32)
            # zero = tf.zeros((batch_size, h, w, 3), dtype=tf.int32)
            model = upsampler.SpatialUpsampler(config.model)
            # model(zero, inputs_slice=zero_slice, training=True)
        ## optimixer
        # decayed_lr = tf.train.exponential_decay(
        #     config.optimizer.learning_rate, global_step, 10000, 0.99, staircase=True)

        # optimizer = tf.train.RMSPropOptimizer(config.optimizer.learning_rate)
        # optimizer = tf.train.AdamOptimizer(decayed_lr)
        # optimizer = tf.train.GradientDescentOptimizer(decayed_lr)
        loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=100,
                                                               decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        # loss_scale_manager = FixedLossScaleManager(2**16)
        opt_tmp = tf.train.RMSPropOptimizer(config.optimizer.learning_rate)
        # opt_tmp = tf.train.AdadeltaOptimizer(config.optimizer.learning_rate)
        # opt_tmp = tf.train.AdamOptimizer(config.optimizer.learning_rate)
        # opt_tmp = tf.train.GradientDescentOptimizer(config.optimizer.learning_rate)
        # opt_tmp = tf.train.MomentumOptimizer(config.optimizer.learning_rate, momentum=0.9)
        optimizer = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)

        loss = loss_on_batch(data_iterator, model, config, training=True)
        tf.add_to_collection('total_loss', loss)
        loss = tf.add_n(tf.get_collection('total_loss'))
        grads = optimizer.compute_gradients(loss)
        # automade loss scale
        # scale = 2**16
        # grads = optimizer.compute_gradients(loss*scale)
        # grads = [(grad / scale, var) for grad, var in grads]
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(0.99, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        saver = tf.train.Saver(tf.global_variables())
        lossT = tf.placeholder(tf.float32)
        lossSumT = tf.summary.scalar('TrnLoss', lossT)
        merged = tf.summary.merge_all()
        sess_config = tf.ConfigProto()
        custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        # set precision mode allow_mix_precision allow_fp32_to_fp16 force_fp32
        custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes(
            'allow_mix_precision')
        # # custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
        # # dump path
        # custom_op.parameter_map['dump_path'].s = tf.compat.as_bytes('/cache/saveModels/')
        # # set dump debug
        # custom_op.parameter_map['enable_dump_debug'].b = True
        # custom_op.parameter_map['dump_debug_mode'].s = tf.compat.as_bytes('all')
        # custom_op.parameter_map["profiling_mode"].b = True
        # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
        #     '{"output":"/cache/saveModels/","task_trace":"on"}')
        # custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes(
            # "/home/ma-user/modelarts/user-job-dir/code/precision_tool/ops_info.json")
        # custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes(
            # '/home/ma-user/modelarts/user-job-dir/code/precision_tool/fusion_switch.cfg')
        sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        # sess_config = npu_tf_config.session_dump_config(sess_config, action='overflow')
        # with tf.Session(config=npu_config_proto(config_proto=sess_config)) as sess:
        with tf.Session(config=sess_config) as sess:
            sess.run(iterator.initializer)
            sess.run(tf.global_variables_initializer())
            if FLAGS.restore:
                ckpt = tf.train.get_checkpoint_state(FLAGS.prelogdir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print('restore model from {}!'.format(FLAGS.pretrain_dir))
            summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
            avgloss = 0.0
            checkpoint_path = os.path.join(FLAGS.logdir, 'model.ckpt')
            for i in range(config.get('max_train_steps', 1000000)):
                start_time = time.time()
                # _, loss_value = sess.run([train_op, loss])
                _, loss_value, scale_value = sess.run([train_op, loss, loss_scale_manager.get_loss_scale()])
                duration = time.time() - start_time
                avgloss = avgloss + loss_value
                if i % 100 == 100 - 1:
                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration
                    format_str = ('%s: step: %d to %d, Avgloss = %.6f (%.1f examples/sec; %.3f '
                                  's), loss_scale = %.2f')
                    print(format_str % (datetime.now(), i - 99, i, avgloss / 100.0,
                                        examples_per_sec, sec_per_batch, scale_value))
                    summary_str = sess.run(merged, feed_dict={lossT: avgloss / 100.0})
                    summary_writer.add_summary(summary_str, i)
                    avgloss = 0.0
                if i % config.save_checkpoint_secs == config.save_checkpoint_secs - 1:
                    checkpoint_path = os.path.join(FLAGS.logdir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=i, write_meta_graph=True)
            saver.save(sess, checkpoint_path, global_step=i, write_meta_graph=True)
            summary_writer.close()
    print('Training completed')
    print('****************************************************')
    # copy results to obs
    # mx.file.copy_parallel('/cache/saveModels', FLAGS.train_url)
    # print('copy saved model to obs: {}.'.format(FLAGS.train_url))


if __name__ == '__main__':
    train()
