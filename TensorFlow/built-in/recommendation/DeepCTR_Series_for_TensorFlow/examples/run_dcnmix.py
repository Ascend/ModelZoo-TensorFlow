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

import tensorflow as tf
tf.enable_control_flow_v2()
tf.enable_resource_variables()

from tensorflow.python.ops.parsing_ops import FixedLenFeature
from deepctr.estimator import DCNMixEstimator
from deepctr.estimator.inputs import input_fn_tfrecord_adv

from npu_bridge.npu_init import *

import argparse
import os
import time

TRAIN_LINE_COUNT = 45840617
CAT_TO_VOC = {
    'C1': 1462,
    'C2': 585,
    'C3': 10131228,
    'C4': 2202609,
    'C5': 307,
    'C6': 25,
    'C7': 12519,
    'C8': 635,
    'C9': 5,
    'C10': 93147,
    'C11': 5685,
    'C12': 8351594,
    'C13': 3196,
    'C14': 29,
    'C15': 14994,
    'C16': 5461307,
    'C17': 12,
    'C18': 5654,
    'C19': 2174,
    'C20': 5,
    'C21': 7046548,
    'C22': 19,
    'C23': 17,
    'C24': 286182,
    'C25': 106,
    'C26': 142573
}

class ExamplesPerSecondHook(tf.train.SessionRunHook):
    def __init__(self, batch_size, iterations_per_loop=1):
        self._batch_size = batch_size
        self._iter_per_loop = iterations_per_loop
        self.start_time = 0
        self.end_time = 0

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self.start_time = time.time()
        return tf.estimator.SessionRunArgs(fetches=[tf.compat.v1.train.get_global_step(), 'add_1:0'])

    def after_run(self, run_context, run_values):
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        global_step, total_loss = run_values.results
        global_step_per_sec = self._iter_per_loop / elapsed_time
        examples_per_sec = self._batch_size * global_step_per_sec
        tf.compat.v1.logging.info('loss = %.7f, step = %d', total_loss, global_step)
        tf.compat.v1.logging.info('global_step/sec: %g', global_step_per_sec)
        tf.compat.v1.logging.info('examples/sec: %g', examples_per_sec)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="./",
                        help='data path for train')
    parser.add_argument('--precision_mode', default='allow_mix_precision',
                        help='allow_fp32_to_fp16/force_fp16/ '
                             'must_keep_origin_dtype/allow_mix_precision.')
    parser.add_argument('--train_batch_size', default=128, type=int,
                        help='batch size of train')
    parser.add_argument('--eval_batch_size', default=128, type=int,
                        help='batch size of eval')
    parser.add_argument('--num_epochs', default=1, type=int,
                        help='train epoch number')
    parser.add_argument('--max_steps', default=None, type=int,
                        help='max train steps')
    parser.add_argument('--iterations_per_loop', default=1, type=int,
                        help='one session run steps')
    parser.add_argument('--output_dir', default="./",
                        help='model path for train')
    args = parser.parse_args()

    # 1.generate feature_column for linear part and dnn part

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    dnn_feature_columns = []
    linear_feature_columns = []

    for i, feat in enumerate(sparse_features):
        dnn_feature_columns.append(tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(feat, CAT_TO_VOC[feat]), 16))
        linear_feature_columns.append(tf.feature_column.categorical_column_with_identity(feat, CAT_TO_VOC[feat]))
    for feat in dense_features:
        dnn_feature_columns.append(tf.feature_column.numeric_column(feat))
        linear_feature_columns.append(tf.feature_column.numeric_column(feat))

    # 2.generate input data for model

    feature_description = {k: FixedLenFeature(dtype=tf.int64, shape=1) for k in sparse_features}
    feature_description.update(
        {k: FixedLenFeature(dtype=tf.float32, shape=1) for k in dense_features})
    feature_description['label'] = FixedLenFeature(dtype=tf.float32, shape=1)

    train_filename_pattern = os.path.join(args.data_path, 'train_part_*.tfrecord')
    train_file_list = sorted(tf.gfile.Glob(train_filename_pattern))
    test_filename_pattern = os.path.join(args.data_path, 'test_part_*.tfrecord')
    test_file_list = sorted(tf.gfile.Glob(test_filename_pattern))

    train_model_input = input_fn_tfrecord_adv(train_file_list, feature_description, 'label',
                                              batch_size=args.train_batch_size,
                                              num_epochs=args.num_epochs, shuffle_factor=10)
    test_model_input = input_fn_tfrecord_adv(test_file_list, feature_description, 'label',
                                             batch_size=args.eval_batch_size,
                                             num_epochs=1, shuffle_factor=0)

    rank_id = int(os.getenv("RANK_ID"))
    rank_size = int(os.getenv("RANK_SIZE"))

    global_batch_size = args.train_batch_size * rank_size

    examples_hook = ExamplesPerSecondHook(global_batch_size, args.iterations_per_loop)

    steps_per_epoch = TRAIN_LINE_COUNT * 0.9 // global_batch_size + 1

    npu_config = NPURunConfig(
        save_checkpoints_steps=steps_per_epoch if rank_id == 0 else 0,
        log_step_count_steps=None,
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False),
        precision_mode=args.precision_mode,
        enable_data_pre_proc=True,
        graph_memory_max_size=str(24 * 1024 * 1024 * 1024),
        variable_memory_max_size=str(7 * 1024 * 1024 * 1024),
        hcom_parallel=True
    )

    # 3.Define Model,train,predict and evaluate
    model = DCNMixEstimator(linear_feature_columns, dnn_feature_columns, cross_num=3, low_rank=256,
                            dnn_hidden_units=(256, 128, 64), dnn_dropout=0.5,
                            model_dir=args.output_dir,
                            config=npu_config, training_chief_hooks=[examples_hook])

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    model.train(train_model_input, max_steps=args.max_steps)
    if rank_id == 0:
        model.evaluate(test_model_input)

if __name__ == "__main__":
    main()


