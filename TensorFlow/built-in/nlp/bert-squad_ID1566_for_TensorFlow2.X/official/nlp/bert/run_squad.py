# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Run BERT on SQuAD 1.1 and SQuAD 2.0 in TF 2.x."""

import json
import os
import time

import npu_device


# Import libraries
from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf
from official.common import distribute_utils
from official.nlp.bert import configs as bert_configs
from official.nlp.bert import run_squad_helper
from official.nlp.bert import tokenization
from official.nlp.data import squad_lib as squad_lib_wp
from official.utils.misc import keras_utils


flags.DEFINE_string('vocab_file', None,
                    'The vocabulary file that the BERT model was trained on.')
flags.DEFINE_integer(name='train_steps', default=0,
                    help='train steps')
flags.DEFINE_boolean(name='use_fastgelu', default=True,
                    help='whether to enable fastgelu, default is True')
flags.DEFINE_string(name='precision_mode', default= 'allow_fp32_to_fp16',
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
flags.DEFINE_boolean(name='use_mixlist', default=False,
                    help='whether to enable mixlist, default is False')
flags.DEFINE_boolean(name='fusion_off_flag', default=False,
                    help='whether to enable fusion_off_flag, default is False')
flags.DEFINE_string(name='mixlist_file', default='ops_info.json',
                    help='mixlist file name, default is ops_info.json')
flags.DEFINE_string(name='fusion_off_file', default='fusion_switch.cfg',
                    help='fusion_off file name, default is fusion_switch.cfg')
flags.DEFINE_boolean(name='auto_tune', default=False,
                    help='whether to enable auto_tune, default is False')
# More flags can be found in run_squad_helper.
run_squad_helper.define_common_squad_flags()

FLAGS = flags.FLAGS

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
                        "fp_point":"", \
                        "bp_point":""}'
    npu_device.global_options().profiling_config.profiling_options = profiling_options
  npu_device.global_options().precision_mode=FLAGS.precision_mode
  if FLAGS.use_mixlist and FLAGS.precision_mode=='allow_mix_precision':
    npu_device.global_options().modify_mixlist=FLAGS.mixlist_file
  if FLAGS.fusion_off_flag:
    npu_device.global_options().fusion_switch_file=FLAGS.fusion_off_file
  if FLAGS.auto_tune:
    npu_device.global_options().auto_tune_mode="RL,GA"
  npu_device.open().as_default()

def train_squad(strategy,
                input_meta_data,
                custom_callbacks=None,
                run_eagerly=False,
                init_checkpoint=None,
                sub_model_export_name=None):
  """Run bert squad training."""
  bert_config = bert_configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  init_checkpoint = init_checkpoint or FLAGS.init_checkpoint
  run_squad_helper.train_squad(strategy, input_meta_data, bert_config,
                               custom_callbacks, run_eagerly, init_checkpoint,
                               sub_model_export_name=sub_model_export_name)


def predict_squad(strategy, input_meta_data):
  """Makes predictions for the squad dataset."""
  bert_config = bert_configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  run_squad_helper.predict_squad(
      strategy, input_meta_data, tokenizer, bert_config, squad_lib_wp)


def eval_squad(strategy, input_meta_data):
  """Evaluate on the squad dataset."""
  bert_config = bert_configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  eval_metrics = run_squad_helper.eval_squad(
      strategy, input_meta_data, tokenizer, bert_config, squad_lib_wp)
  return eval_metrics


def export_squad(model_export_path, input_meta_data):
  """Exports a trained model as a `SavedModel` for inference.

  Args:
    model_export_path: a string specifying the path to the SavedModel directory.
    input_meta_data: dictionary containing meta data about input and model.

  Raises:
    Export path is not specified, got an empty string or None.
  """
  bert_config = bert_configs.BertConfig.from_json_file(FLAGS.bert_config_file)
  run_squad_helper.export_squad(model_export_path, input_meta_data, bert_config)


def main(_):
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  npu_config()

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  if FLAGS.mode == 'export_only':
    export_squad(FLAGS.model_export_path, input_meta_data)
    return

  # Configures cluster spec for multi-worker distribution strategy.
  if FLAGS.num_gpus > 0:
    _ = distribute_utils.configure_cluster(FLAGS.worker_hosts, FLAGS.task_index)
  strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      all_reduce_alg=FLAGS.all_reduce_alg,
      tpu_address=FLAGS.tpu)

  if 'train' in FLAGS.mode:
    if FLAGS.log_steps:
      custom_callbacks = [keras_utils.TimeHistory(
          batch_size=FLAGS.train_batch_size,
          log_steps=FLAGS.log_steps,
          logdir=FLAGS.model_dir,
      )]
    else:
      custom_callbacks = None

    train_squad(
        strategy,
        input_meta_data,
        custom_callbacks=custom_callbacks,
        run_eagerly=FLAGS.run_eagerly,
        sub_model_export_name=FLAGS.sub_model_export_name,
    )
  if 'predict' in FLAGS.mode:
    predict_squad(strategy, input_meta_data)
  if 'eval' in FLAGS.mode:
    eval_metrics = eval_squad(strategy, input_meta_data)
    f1_score = eval_metrics['final_f1']
    logging.info('SQuAD eval F1-score: %f', f1_score)
    summary_dir = os.path.join(FLAGS.model_dir, 'summaries', 'eval')
    summary_writer = tf.summary.create_file_writer(summary_dir)
    with summary_writer.as_default():
      # TODO(lehou): write to the correct step number.
      tf.summary.scalar('F1-score', f1_score, step=0)
      summary_writer.flush()
    # Also write eval_metrics to json file.
    squad_lib_wp.write_to_json_files(
        eval_metrics, os.path.join(summary_dir, 'eval_metrics.json'))
    time.sleep(60)


if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
