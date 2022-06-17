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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# os.system("pip install colorlog")
# os.system("pip install h5py==2.9.0")


from npu_bridge.npu_init import *
import datasets.hdf5_loader as dataset
import time

from six.moves import xrange
from pprint import pprint
import h5py
import tensorflow as tf
import tensorflow.contrib.slim as slim

from input_ops import create_input_ops,my_create_input_ops,hdf2tfrecord
from util import log
from config import argparser
# import moxing as mox
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

class Trainer(object):

    def __init__(self, config, model, dataset, dataset_test):
        self.config = config
        self.model = model
        hyper_parameter_str = '{}_lr_g_{}_d_{}_update_G{}D{}'.format(
            config.dataset, config.learning_rate_g, config.learning_rate_d, 
            config.update_rate, 1
        )
        self.train_dir = '/cache/train_dir/%s-%s-%s' % (
            config.prefix,
            hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )

        os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        # _, self.batch_train = create_input_ops(
        #     dataset, self.batch_size, is_training=True)
        # _, self.batch_test = create_input_ops(
        #     dataset_test, self.batch_size, is_training=False)


        TRAIN_TFRECORD_PATH = "/cache/data/train-tfrecords"
        TEST_TFRECORD_PATH = "/cache/data/test-tfrecords"

        hdf2tfrecord(TRAIN_TFRECORD_PATH,is_train=True)
        hdf2tfrecord(TEST_TFRECORD_PATH,is_train=False)

        _, self.batch_train,self.train_iter = my_create_input_ops(TRAIN_TFRECORD_PATH,self.batch_size)
        _, self.batch_test,self.test_iter = my_create_input_ops(TEST_TFRECORD_PATH, self.batch_size)


        # --- optimizer ---
        self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)

        # --- checkpoint and monitoring ---
        all_var = tf.trainable_variables()

        d_var = [v for v in all_var if v.name.startswith('Discriminator')]
        log.warn("********* d_var ********** ")
        slim.model_analyzer.analyze_vars(d_var, print_info=True)

        g_var = [v for v in all_var if v.name.startswith(('Generator'))]
        log.warn("********* g_var ********** ")
        slim.model_analyzer.analyze_vars(g_var, print_info=True)

        rem_var = (set(all_var) - set(d_var) - set(g_var))
        print([v.name for v in rem_var])
        assert not rem_var

        self.d_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.d_loss,
            global_step=self.global_step,
            learning_rate=self.config.learning_rate_d,
            optimizer=tf.train.AdamOptimizer(beta1=0.5),
            clip_gradients=20.0,
            name='d_optimize_loss',
            variables=d_var
        )

        self.g_optimizer = tf.contrib.layers.optimize_loss(
            loss=self.model.g_loss,
            global_step=self.global_step,
            learning_rate=self.config.learning_rate_g,
            optimizer=tf.train.AdamOptimizer(beta1=0.5),
            clip_gradients=20.0,
            name='g_optimize_loss',
            variables=g_var
        )

        self.summary_op = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=1000)
        self.summary_writer = tf.summary.FileWriter(self.train_dir)

        self.supervisor = tf.train.Supervisor(
            logdir=self.train_dir,
            is_chief=True,
            saver=None,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_summaries_secs=300,
            save_model_secs=600,
            global_step=self.global_step,
        )

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            #gpu_options=tf.GPUOptions(allow_growth=True),
            device_count={'NPU': 1}
        )
        # session_config = tf.ConfigProto(allow_soft_placement=True)

        # 手动迁移
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        #custom_op.parameter_map["use_off_line"].b = True    #PASS
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap

        # custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes("allow_mix_precision")
        # session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
        # session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)

        self.ckpt_path = config.checkpoint
        if self.ckpt_path is not None:
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.saver.restore(self.session, self.ckpt_path)
            log.info("Loaded the pretrain parameters from the provided checkpoint path")

    def train(self):
        log.infov("Training Starts!")
        pprint(self.batch_train)
        step = self.session.run(self.global_step)
        self.session.run(self.train_iter.initializer)
        self.session.run(self.test_iter.initializer)
        for s in xrange(self.config.max_training_steps):

            # periodic inference
            if s % self.config.test_sample_step == 0:
                accuracy, d_loss, g_loss, s_loss, step_time = \
                    self.run_test(self.batch_test, is_train=False)
                self.log_step_message(step, accuracy, d_loss, g_loss,
                                      s_loss, step_time, is_train=False)
           
            step, accuracy, summary, d_loss, g_loss, s_loss, step_time, prediction_train, gt_train, g_img = \
                self.run_single_step(self.batch_train, step=s)

            if s % self.config.log_step == 0:
                self.log_step_message(step, accuracy,  d_loss, g_loss, s_loss, step_time)

            if s % self.config.write_summary_step == 0:
                self.summary_writer.add_summary(summary, global_step=step)

            if s % self.config.output_save_step == 0:
                log.infov("Saved checkpoint at %d", step)
                save_path = self.saver.save(self.session, os.path.join(self.train_dir, 'model'), global_step=step)
                if self.config.dump_result:
                    f = h5py.File(os.path.join(self.train_dir, 'g_img_'+str(s)+'.hdf5'), 'w')
                    f['image'] = g_img
                    f.close()

        self.session.close()

    def run_single_step(self, batch, step=None, is_train=True):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        fetch = [self.global_step, self.model.accuracy, self.summary_op, 
                 self.model.d_loss, self.model.g_loss, self.model.S_loss, 
                 self.model.all_preds, self.model.all_targets, 
                 self.model.fake_image]

        if step % (self.config.update_rate+1) > 0:
        # Train the generator
            fetch.append(self.g_optimizer)
        else:
        # Train the discriminator
            fetch.append(self.d_optimizer)

        fetch_values = self.session.run(fetch,
            feed_dict=self.model.get_feed_dict(batch_chunk, step=step)
        )

        [step, loss, summary, d_loss, g_loss, \
         s_loss, all_preds, all_targets, g_img] = fetch_values[:9]

        _end_time = time.time()

        return step, loss, summary, d_loss, g_loss, s_loss, \
            (_end_time - _start_time), all_preds, all_targets, g_img

    def run_test(self, batch, is_train=False):
        _start_time = time.time()

        batch_chunk = self.session.run(batch)

        [accuracy, d_loss, g_loss, s_loss] = self.session.run(
            [self.model.accuracy, self.model.d_loss, 
             self.model.g_loss, self.model.S_loss],
            feed_dict=self.model.get_feed_dict(batch_chunk, is_training=False))

        _end_time = time.time()

        return accuracy, d_loss, g_loss, s_loss, (_end_time - _start_time)

    def log_step_message(self, step, accuracy, d_loss, g_loss, 
                         s_loss, step_time, is_train=True):
        if step_time == 0: step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Supervised loss: {s_loss:.5f} " +
                "D loss: {d_loss:.5f} " +
                "G loss: {g_loss:.5f} " +
                "Accuracy: {accuracy:.5f} "
                "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) "
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step = step,
                         d_loss = d_loss,
                         g_loss = g_loss,
                         s_loss = s_loss,
                         accuracy = accuracy,
                         sec_per_batch = step_time,
                         instance_per_sec = self.batch_size / step_time
                         )
               )

def main():
    # obs://improvedgan/datasets/datasets/

    # mox.file.copy_parallel(
        # 'obs://improvedgan/datasets/datasets',
        # '/cache/data')

    # store
    filename = "/cache/train_dir"
    if not os.path.exists(filename):
    	os.mkdir(filename)

    config, model, dataset_train, dataset_test = argparser(is_train=True)

    trainer = Trainer(config, model, dataset_train, dataset_test)

    log.warning("dataset: %s, learning_rate_g: %f, learning_rate_d: %f",
                config.dataset, config.learning_rate_g, config.learning_rate_d)
    trainer.train()

    # from .help_modelarts import modelarts_result2obs
    # class FLAG():
    #     def __init__(self):
    #         self.obs_dir = ""
    #         self.log_dir = ""
    #
    #
    # FLAGS = FLAG()
    # modelarts_result2obs(FLAGS)

    # mox.file.copy_parallel('/cache/train_dir',
        # 'obs://improvedgan/train_dir')

if __name__ == '__main__':
    main()