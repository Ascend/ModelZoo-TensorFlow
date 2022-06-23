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
"""Script to evaluate a trained Attention OCR model.

A simple usage example:
python eval.py
"""
# from npu_bridge.npu_init import *
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.compat.v1 import flags
from tensorflow.python.training import checkpoint_management
import time

import data_provider
import common_flags
import os

from tensorflow.python.platform import tf_logging as logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = flags.FLAGS
common_flags.define()

# yapf: disable

#100
flags.DEFINE_integer('num_batches', 1339,
                     'Number of batches to run eval for.')

flags.DEFINE_string('eval_log_dir', '/tmp/attention_ocr/eval',
                    'Directory where the evaluation results are saved to.')

flags.DEFINE_integer('eval_interval_secs', 60,
                     'Frequency in seconds to run evaluations.')

flags.DEFINE_integer('number_of_steps', 1,
                     'Number of times to run evaluation.')

flags.DEFINE_bool('my_eval', False,
                     'use my eval.')
flags.DEFINE_bool('Not_on_modelart', True,
                     'use my eval.')

# yapf: enable

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def my_eval():
    if not tf.io.gfile.exists(FLAGS.eval_log_dir):
        tf.io.gfile.makedirs(FLAGS.eval_log_dir)

    dataset = common_flags.create_eval_dataset(split_name='test')
    model = common_flags.create_model(dataset.num_char_classes,
                                      dataset.max_sequence_length,
                                      dataset.num_of_views, dataset.null_code)
    data = data_provider.get_data(
        dataset,
        FLAGS.batch_size,
        augment=False,
        central_crop_size=common_flags.get_crop_size(),
        shuffle=False)

    endpoints = model.create_base(data.images, labels_one_hot=None)
    model.create_loss(data, endpoints)
    eval_ops = model.create_summaries(
        data, endpoints, dataset.charset, is_training=False)


    slim.get_or_create_global_step()
    session_config = tf.compat.v1.ConfigProto(device_count={"GPU": 2})
    ckpt_dir = FLAGS.train_log_dir
    ckpt_dir = '/home/zixuan/attention_ocr/my_ckpt'
    ckpt_dir = '/tmp/attention_ocr/train'

    logdir = '/tmp/attention_ocr/my_test'

    if not tf.io.gfile.exists(logdir):
        logging.info('Create a new training directory %s', logdir)
        tf.io.gfile.makedirs(logdir)

    ckpt_number=400000
    while True:
        saver = tf.train.Saver(max_to_keep=5)
        with tf.Session() as session:
            checkpoint_dir = logdir
            checkpoint_dir = '/home/zixuan/attention_ocr/my_ckpt_6_9'
            ckpt = checkpoint_management.get_checkpoint_state(checkpoint_dir)
            new_path = '/home/zixuan/attention_ocr/my_ckpt_6_9/'+ckpt.model_checkpoint_path.split('/')[-1]
            new_path = '/home/zixuan/attention_ocr/my_ckpt_6_9/'+'model.ckpt-'+str(ckpt_number)
            if ckpt and ckpt.model_checkpoint_path:
                step_index = int(ckpt.model_checkpoint_path.split("-")[-1])
                saver.restore(session, new_path)
            saver.save(session, logdir + os.sep + 'model.ckpt', global_step=ckpt_number)
        ckpt_dir = logdir
        # ckpt_dir = "./my_ckpt"
        slim.evaluation.evaluation_loop(
            master=FLAGS.master,
            # checkpoint_dir=FLAGS.train_log_dir,
            checkpoint_dir=ckpt_dir,
            logdir=FLAGS.eval_log_dir,
            eval_op=eval_ops,
            num_evals=FLAGS.num_batches,
            eval_interval_secs=FLAGS.eval_interval_secs,
            max_number_of_evaluations=1,
            session_config=session_config)

        del_file(logdir)


def main(_):
    if not tf.io.gfile.exists(FLAGS.eval_log_dir):
        tf.io.gfile.makedirs(FLAGS.eval_log_dir)

    if FLAGS.my_eval == True:
        my_eval()

    dataset = common_flags.create_eval_dataset(split_name='test')
    model = common_flags.create_model(dataset.num_char_classes,
                                      dataset.max_sequence_length,
                                      dataset.num_of_views, dataset.null_code)
    data = data_provider.get_data(
        dataset,
        FLAGS.batch_size,
        augment=False,
        central_crop_size=common_flags.get_crop_size(),
        shuffle=False)

    endpoints = model.create_base(data.images, labels_one_hot=None)
    model.create_loss(data, endpoints)
    eval_ops = model.create_summaries(
        data, endpoints, dataset.charset, is_training=False)


    slim.get_or_create_global_step()
    session_config = tf.compat.v1.ConfigProto(device_count={"GPU": 0}, allow_soft_placement=True)
    if FLAGS.not_on_modelart == True:
        ckpt_dir = FLAGS.train_log_dir
    else:
        ckpt_dir = "./ckpt"
    print("Start Evaluating!")
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        # checkpoint_dir=FLAGS.train_log_dir,
        checkpoint_dir=ckpt_dir,
        logdir=FLAGS.eval_log_dir,
        eval_op=eval_ops,
        num_evals=FLAGS.num_batches,
        eval_interval_secs=FLAGS.eval_interval_secs,
        max_number_of_evaluations=FLAGS.number_of_steps,
        session_config=session_config)



if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    app.run()

