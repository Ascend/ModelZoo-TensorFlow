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

# pylint: disable=logging-format-interpolation
# pylint: disable=unused-import
# pylint: disable=g-long-lambda
# pylint: disable=g-direct-tensorflow-import
# pylint: disable=line-too-long

r"""Docs."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function
from npu_bridge.npu_init import *
#import precision_tool.tf_config as npu_tf_config
#import precision_tool.config as npu_common_config

import collections
import itertools
import json
import os
import sys
import threading
import time
import traceback

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm

import common_utils
import data_utils
import flag_utils
import modeling
import training_utils
from training_utils import eval_step_fn, eval_accuracy, eval_ema_step_fn
from data_utils import get_eval_size

os.environ["TF_CPP_VMODULE"] = "auto_mixed_precision=2"
FLAGS = flags.FLAGS
# Fusion switch file path
FUSION_SWITCH_FILE = os.path.join(os.path.dirname(__file__), 'fusion_switch.cfg')
OPS_FILE = os.path.join(os.path.dirname(__file__), 'ops_info.json')

if 'gfile' not in sys.modules:
  gfile = tf.gfile


def get_model_builder(params):
  """Return the function that builds models."""
  if params.model_type == 'wrn-28-2':
    return modeling.Wrn28k(params, k=2)
  elif params.model_type == 'wrn-28-10':
    return modeling.Wrn28k(params, k=10)
  elif params.model_type == 'wrn-28-large':
    return modeling.Wrn28k(params, k=135)
  elif params.model_type == 'resnet-50':
    return modeling.ResNet50(params)
  elif params.model_type.startswith('efficientnet-'):
    model = modeling.EfficientNet(params)
    if 'imagenet' in params.dataset_name.lower():
      params['eval_image_size'] = model.eval_image_size
    return model
  elif params.model_type.startswith('nas-'):
    model = modeling.CustomNet(params)
    if 'imagenet' in params.dataset_name.lower():
      params['eval_image_size'] = model.eval_image_size
    return model
  else:
    raise ValueError(f'Unknown model_type `{params.model_type}`')


def train_gpu(params):
    ## Setup session
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    config_proto.allow_soft_placement = True
    # added for enabling mix precision and loss scale
    config_proto.graph_options.rewrite_options.auto_mixed_precision = 1
    #config_proto = tf.ConfigProto(device_count={'GPU':0})

    # cann used
    custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config_proto.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    # 设置混合精度
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    # 设置混合精度黑名单
    #custom_op = npu_tf_config.update_custom_op_adv(custom_op, action="modify_mixlist")
    custom_op.parameter_map['modify_mixlist'].s = tf.compat.as_bytes(OPS_FILE)
    print("[PrecisionTool] Set mix_precision setting file: ", OPS_FILE)

    # 打开浮点溢出开关
    #custom_op = npu_tf_config.update_custom_op_adv(custom_op, action='overflow')
    ## 设置算子融合规则
    #custom_op = npu_tf_config.update_custom_op_adv(custom_op, action='fusion_switch')
    custom_op.parameter_map['fusion_switch_file'].s = tf.compat.as_bytes(FUSION_SWITCH_FILE)
    print("[PrecisionTool] Set fusion switch file: ", FUSION_SWITCH_FILE)

    # custom_op = npu_tf_config.update_custom_op_adv(custom_op, action='fusion_off')
    # config_proto = npu_tf_config.session_dump_config(config_proto, action='fusion_off')  # 关闭所有融合规则
    tfs = tf.Session(config=npu_config_proto(config_proto=config_proto))

    # train dataset with labeled and unlabeled
    train_dataset = data_utils.build_train_dataset(
        params=params,
        batch_size=params.train_batch_size,
        num_inputs=1,
        input_index=0)
    train_data_inter = train_dataset.make_initializable_iterator()
    tfs.run(train_data_inter.initializer)
    l_images, l_labels, u_images_ori, u_images_aug = train_data_inter.get_next()

    # test dataset
    test_dataset = data_utils.build_eval_dataset(
        params=params,
        batch_size=params.eval_batch_size,
        num_workers=1,
        worker_index=0)
    test_data_inter = test_dataset.make_initializable_iterator()
    tfs.run(test_data_inter.initializer)
    test_images, test_labels = test_data_inter.get_next()

    # train operation
    model = get_model_builder(params)
    mpl_model = training_utils.MPL()
    merged_summary_op, teacher_train_op, train_t_logits, train_t_labels = \
        mpl_model.step_fn(params, model, l_images, l_labels, u_images_ori, u_images_aug)

    # test operation
    student_logits, ema_logits, teacher_logits, student_loss, ema_loss, teacher_loss = eval_step_fn(params, model, test_images, test_labels)

    # Create model saver
    saver = tf.train.Saver(max_to_keep=6)
    model_path = os.path.join(params.output_path, "ckpts/model.ckpt")
    latest_model_dir = os.path.join(params.output_path, "ckpts/")
    saver_ema = tf.train.Saver(max_to_keep=6)
    model_path_ema = os.path.join(params.output_path, "ema_ckpts/model.ckpt")
    latest_model_dir_ema = os.path.join(params.output_path, "ema_ckpts/")

    # Create summary writer
    train_log_path = os.path.join(params.output_path, "train_logs")
    train_summary_writer = tf.summary.FileWriter(train_log_path, tfs.graph)
    test_log_path = os.path.join(params.output_path, "test_logs")
    test_summary_writer = tf.summary.FileWriter(test_log_path)

    # Training
    global_step = tf.train.get_or_create_global_step()
    tfs.run(tf.global_variables_initializer())
    tfs.run(tf.local_variables_initializer())
    num_train_steps = params.num_train_steps
    num_eval_steps = get_eval_size(params) // params.eval_batch_size
    if params.load_ema_checkpoint:
        latest_ckpt = tf.train.latest_checkpoint(latest_model_dir_ema)
        saver_ema.restore(tfs, latest_ckpt)
        logging.info("Restore from ema ckpt:{}".format(latest_ckpt))
    elif params.load_checkpoint:
        latest_ckpt = tf.train.latest_checkpoint(latest_model_dir)
        saver.restore(tfs, latest_ckpt)
        logging.info("Restore from student ckpt:{}".format(latest_ckpt))

    global_inter = tfs.run(global_step)
    # debug: lr_temp = tf.train.cosine_decay(learning_rate=0.0125, global_step=-7940, decay_steps=292000, alpha=0.0)
    train_cost_list = list()
    max_top1, max_top1_ema = 0.0, 0.0

    while global_inter < num_train_steps:
        train_start_time = time.time()
        merged_summary, _, train_t_logits_value, train_t_labels_value \
            = tfs.run([merged_summary_op, teacher_train_op, train_t_logits, train_t_labels])
        train_top_1_num, train_top_5_num = eval_accuracy(train_t_labels_value, train_t_logits_value)
        train_top1 = train_top_1_num / params.train_batch_size
        train_top5 = train_top_5_num / params.train_batch_size
        test_summary_writer.add_summary(
            tf.Summary(value=[tf.Summary.Value(tag='eval/train_teacher_top1', simple_value=train_top1), tf.Summary.Value(tag='eval/train_teacher_top5', simple_value=train_top5)]), global_inter)
        test_summary_writer.flush()

        train_cost_list.append(time.time() - train_start_time)
        train_summary_writer.add_summary(merged_summary, global_inter)
        if ((global_inter + 1) % params.log_every) == 0:
            start_step = global_inter - len(train_cost_list) + 1
            logging.info("Train total cost in global steps {}-{}:{}".format(start_step, global_inter, sum(train_cost_list)))
            train_cost_list.clear()
        if ((global_inter + 1) % params.save_every) == 0:
            # Eval op
            s_top_1, s_top_5 = 0.0, 0.0
            ema_top_1, ema_top_5 = 0.0, 0.0
            t_top_1, t_top_5 = 0.0, 0.0
            student_avg_loss, ema_avg_loss, teacher_avg_loss = 0.0, 0.0, 0.0
            logging.info("Eval start in global step:{}".format(global_inter))
            eval_start_time = time.time()
            for eval_step in range(num_eval_steps):
                test_labels_value, student_logits_value, ema_logits_value, teacher_logits_value, student_loss_value, ema_loss_value, teacher_loss_value = \
                        tfs.run([test_labels, student_logits, ema_logits, teacher_logits, student_loss, ema_loss, teacher_loss])
                top_1_num, top_5_num = eval_accuracy(test_labels_value, teacher_logits_value)
                t_top_1 += top_1_num
                t_top_5 += top_5_num
                top_1_num, top_5_num = eval_accuracy(test_labels_value, student_logits_value)
                s_top_1 += top_1_num
                s_top_5 += top_5_num
                top_1_num, top_5_num = eval_accuracy(test_labels_value, ema_logits_value)
                ema_top_1 += top_1_num
                ema_top_5 += top_5_num
                teacher_avg_loss += teacher_loss_value
                student_avg_loss += student_loss_value
                ema_avg_loss += ema_loss_value
            eval_cost = time.time() - eval_start_time
            total_samples = num_eval_steps * params.eval_batch_size
            t_top_1 /= total_samples
            t_top_5 /= total_samples
            s_top_1 /= total_samples
            s_top_5 /= total_samples
            ema_top_1 /= total_samples
            ema_top_5 /= total_samples
            teacher_avg_loss /= num_eval_steps
            student_avg_loss /= num_eval_steps
            ema_avg_loss /= num_eval_steps
            print("Eval end in global step:{}, cost= {}".format(global_inter, eval_cost))
            print("Eval end in global step:{}, teacher_top1={}, teacher_top5={}, teacher_loss = {}".format(global_inter, t_top_1, t_top_5, teacher_avg_loss))
            print("Eval end in global step:{}, student_top1={}, student_top5={}, student_loss = {}".format(global_inter, s_top_1, s_top_5, student_avg_loss))
            print("Eval end in global step:{}, eemmmaa_top1={}, eemmmaa_top5={}, eemmmaa_loss = {}".format(global_inter, ema_top_1, ema_top_5, ema_avg_loss))
            test_summary_writer.add_summary(
                tf.Summary(value=[tf.Summary.Value(tag='eval/teacher_top1', simple_value=t_top_1), tf.Summary.Value(tag='eval/teacher_top5', simple_value=t_top_5),
                                  tf.Summary.Value(tag='eval/student_top1', simple_value=s_top_1), tf.Summary.Value(tag='eval/student_top5', simple_value=s_top_5),
                                  tf.Summary.Value(tag='eval/ema_top1', simple_value=ema_top_1), tf.Summary.Value(tag='eval/ema_top5', simple_value=ema_top_5),
                                  tf.Summary.Value(tag='eval/student_loss', simple_value=student_avg_loss),
                                  tf.Summary.Value(tag='eval/ema_loss', simple_value=ema_avg_loss)]), global_inter)
            test_summary_writer.flush()

            # Save model weights to disk
            if (s_top_1 > max_top1 + 0.1) or (s_top_1 > max_top1 and s_top_1 >= 0.95):
                max_top1 = s_top_1
                save_path = saver.save(tfs, model_path, global_step=global_inter)
                logging.info("Saved Model for student in file:{}, accuracy:{}".format(save_path, max_top1))
            if (ema_top_1 > max_top1_ema + 0.1) or (ema_top_1 > max_top1_ema and ema_top_1 >= 0.95):
                max_top1_ema = ema_top_1
                save_path = saver_ema.save(tfs, model_path_ema, global_step=global_inter)
                logging.info("Saved Model for ema in file:{}, accuracy:{}".format(save_path, max_top1_ema))
        global_inter += 1
        ## debug
        # if global_inter % 5 == 0:
        #     global_step_value = tfs.run(global_step)
        #     logging.info("Global_inter\t{}\tGlobal_step\t{}".format(global_inter, global_step_value))
    test_summary_writer.close()
    tfs.close()


def eval_gpu(params):
    ## Setup session
    config_proto = tf.ConfigProto()
    #config_proto.gpu_options.allow_growth = True
    #config_proto.allow_soft_placement = True
    ## added for enabling mix precision and loss scale
    #config_proto.graph_options.rewrite_options.auto_mixed_precision = 1
    #config_proto = tf.ConfigProto(device_count={'GPU':0})
    tfs = tf.Session(config=npu_config_proto(config_proto=config_proto))

    # test dataset
    test_dataset = data_utils.build_eval_dataset(
        params=params,
        batch_size=params.eval_batch_size,
        num_workers=1,
        worker_index=0)
    test_data_inter = test_dataset.make_initializable_iterator()
    tfs.run(test_data_inter.initializer)
    test_images, test_labels = test_data_inter.get_next()

    # test operation
    model = get_model_builder(params)
    ema_logits = eval_ema_step_fn(params, model, test_images)

    # restore from checkpoint
    saver_ema = tf.train.Saver(max_to_keep=6)
    tfs.run(tf.global_variables_initializer())
    tfs.run(tf.local_variables_initializer())
    num_eval_steps = get_eval_size(params) // params.eval_batch_size
    if params.load_ema_checkpoint:
        latest_ckpt = tf.train.latest_checkpoint(params.output_path)
        saver_ema.restore(tfs, latest_ckpt)
        logging.info("Restore from ema ckpt:{}".format(latest_ckpt))
    else:
        logging.info("Restore checkpoint in failure")
        return

    ema_top_1, ema_top_5 = 0.0, 0.0
    eval_start_time = time.time()
    for eval_step in range(num_eval_steps):
        test_labels_value, ema_logits_value = tfs.run([test_labels, ema_logits])
        top_1_num, top_5_num = eval_accuracy(test_labels_value, ema_logits_value)
        ema_top_1 += top_1_num
        ema_top_5 += top_5_num
    eval_cost = time.time() - eval_start_time
    total_samples = num_eval_steps * params.eval_batch_size
    ema_top_1 /= total_samples
    ema_top_5 /= total_samples
    print("Eval cost in {} seconds; top1 = {}; top5 = {};".format(eval_cost, ema_top_1, ema_top_5))


def freeze_pb_graph(params):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="input")
    model = get_model_builder(params)
    with tf.variable_scope('ema', reuse=True):
      with tf.variable_scope('model'):
        logits = model(inputs, training=False)
        predict_class = tf.nn.softmax(logits, axis=-1, name='output')

    ckpt_dir = params.data_path
    best_ckpt_path = os.path.join(ckpt_dir, 'model.ckpt-274499')
    best_ckpt_meta = os.path.join(ckpt_dir, 'model.ckpt-274499.meta')
    logging.info("Restore from best ckpt:{}".format(best_ckpt_path))

    tfs = tf.Session()
    tfs.run(tf.global_variables_initializer())
    tfs.run(tf.local_variables_initializer())
    saver = tf.train.import_meta_graph(best_ckpt_meta)
    saver.restore(tfs, best_ckpt_path)

    frozen_gd = tf.graph_util.convert_variables_to_constants(
        tfs, tf.get_default_graph().as_graph_def(), ['ema/model/output'])
    pb_model_dir = os.path.join(params.output_path, 'pb_model')
    tf.io.write_graph(frozen_gd, pb_model_dir, "meta_pseudo_labels.pb", as_text=False)


def main(unused_argv):
    try:
        params = flag_utils.build_params_from_flags()
        if params.task_mode == "train":
            train_gpu(params)
        elif params.task_mode == "eval":
            eval_gpu(params)
        elif params.task_mode == "freeze":
            freeze_pb_graph(params)
        else:
            logging.error("Task mode is invalid!!!")
    except:  # pylint: disable=bare-except
        traceback.print_exc()


if __name__ == '__main__':
    tf.disable_v2_behavior()
    np.set_printoptions(precision=3, suppress=True, threshold=int(1e9), linewidth=160)
    app.run(main)    
