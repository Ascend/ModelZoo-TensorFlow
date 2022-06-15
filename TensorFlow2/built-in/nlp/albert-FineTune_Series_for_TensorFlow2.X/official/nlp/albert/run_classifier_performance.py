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

"""ALBERT classification finetuning runner in tf2.x."""

import json
import os
# Import libraries
from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from official.common import distribute_utils
from official.nlp.albert import configs as albert_configs
from official.nlp.bert import bert_models
from official.nlp.bert import run_classifier_performance as run_classifier_bert
import npu_device

flags.DEFINE_string("name='log_steps'", default='12271', 
                        help="TimeHis log Step.")
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

FLAGS = tf.compat.v1.app.flags.FLAGS

def npu_config():
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
  npu_device.global_options().precision_mode = FLAGS.precision_mode
  if FLAGS.use_mixlist and FLAGS.precision_mode=='allow_mix_precision':
    npu_device.global_options().modify_mixlist=FLAGS.mixlist_file
  if FLAGS.fusion_off_flag:
    npu_device.global_options().fusion_switch_file=FLAGS.fusion_off_file
  if FLAGS.auto_tune:
    npu_device.global_options().auto_tune_mode="RL,GA"
  npu_device.open().as_default()
#===============================NPU Migration=========================================
npu_config()

def predict(strategy, albert_config, input_meta_data, predict_input_fn):
  """Function outputs both the ground truth predictions as .tsv files."""
  with strategy.scope():
    classifier_model = bert_models.classifier_model(
        albert_config, input_meta_data['num_labels'])[0]
    checkpoint = tf.train.Checkpoint(model=classifier_model)
    latest_checkpoint_file = (
        FLAGS.predict_checkpoint_path or
        tf.train.latest_checkpoint(FLAGS.model_dir))
    assert latest_checkpoint_file
    logging.info('Checkpoint file %s found and restoring from '
                 'checkpoint', latest_checkpoint_file)
    checkpoint.restore(
        latest_checkpoint_file).assert_existing_objects_matched()
    preds, ground_truth = run_classifier_bert.get_predictions_and_labels(
        strategy, classifier_model, predict_input_fn, return_probs=True)
    output_predict_file = os.path.join(FLAGS.model_dir, 'test_results.tsv')
    with tf.io.gfile.GFile(output_predict_file, 'w') as writer:
      logging.info('***** Predict results *****')
      for probabilities in preds:
        output_line = '\t'.join(
            str(class_probability)
            for class_probability in probabilities) + '\n'
        writer.write(output_line)
    ground_truth_labels_file = os.path.join(FLAGS.model_dir,
                                            'output_labels.tsv')
    with tf.io.gfile.GFile(ground_truth_labels_file, 'w') as writer:
      logging.info('***** Ground truth results *****')
      for label in ground_truth:
        output_line = '\t'.join(str(label)) + '\n'
        writer.write(output_line)
  return


def main(_):
  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  if not FLAGS.model_dir:
    FLAGS.model_dir = '/tmp/bert20/'

  strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      tpu_address=FLAGS.tpu)
  max_seq_length = input_meta_data['max_seq_length']
  train_input_fn = run_classifier_bert.get_dataset_fn(
      FLAGS.train_data_path,
      max_seq_length,
      FLAGS.train_batch_size,
      is_training=True)
  eval_input_fn = run_classifier_bert.get_dataset_fn(
      FLAGS.eval_data_path,
      max_seq_length,
      FLAGS.eval_batch_size,
      is_training=False)

  albert_config = albert_configs.AlbertConfig.from_json_file(
      FLAGS.bert_config_file)
  if FLAGS.mode == 'train_and_eval':
    run_classifier_bert.run_bert(strategy, input_meta_data, albert_config,
                                 train_input_fn, eval_input_fn)
  elif FLAGS.mode == 'predict':
    predict(strategy, albert_config, input_meta_data, eval_input_fn)
  else:
    raise ValueError('Unsupported mode is specified: %s' % FLAGS.mode)
  return

if __name__ == '__main__':
  flags.mark_flag_as_required('bert_config_file')
  flags.mark_flag_as_required('input_meta_data_path')
  flags.mark_flag_as_required('model_dir')
  app.run(main)
