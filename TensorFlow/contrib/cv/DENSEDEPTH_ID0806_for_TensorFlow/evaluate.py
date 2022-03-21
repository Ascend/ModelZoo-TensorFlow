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

import os
import time
import argparse
from npu_bridge.npu_init import *
import tensorflow as tf
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from model import create_model
from utils import evaluate
import utils


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--ckptdir',
                        default=r'./ckpt_npu',
                        type=str, help='Trained Keras model file')
    parser.add_argument('--test_data', type=str, default=r'./dataset/nyu_test.zip',
                        help='test dataset path.')
    parser.add_argument('--bs', default=2, type=int, help='Batch size')
    parser.add_argument('--logdir', type=str, default=r'./', help='evaluation log')
    args = parser.parse_args()
    return args


def ckpt_evaluation(test_data, evaluate_ckptdir, bs, logdir, is_distributed=False):
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["graph_run_mode"].i = 0
    custom_op.parameter_map["op_select_implmode"].s = tf.compat.as_bytes("high_precision")
    # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    npu_keras_sess = set_keras_session_npu_config(config=sess_config)

    start = time.time()

    # Load model into NPU
    model = create_model()

    rgb = test_data['rgb']
    depth = test_data['depth']
    crop = test_data['crop']

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(tf.train.latest_checkpoint(
        evaluate_ckptdir)).assert_existing_objects_matched().expect_partial().run_restore_ops()

    e = evaluate(model, rgb, depth, crop, batch_size=bs, is_distributed=is_distributed)

    end = time.time()

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    with open(os.path.join(logdir, "evaluate.log"), "a+") as f:
        f.write("\n" + evaluate_ckptdir + "\na1 %-10.4f a2 %-10.4f a3 %-10.4f rel %-10.4f rms %-10.4f log_10 "
                                          "%-10.4f" % (e[0], e[1], e[2], e[3], e[4], e[5]))
    print('*' * 20)
    print(evaluate_ckptdir + "\na1 %-10.4f a2 %-10.4f a3 %-10.4f rel %-10.4f rms %-10.4f log_10 %-10.4f "
                             "Test_time %-10.4f" % (
              e[0], e[1], e[2], e[3], e[4], e[5], end - start))
    print('*' * 20)
    close_session(npu_keras_sess)


def main():
    args = parse_args()
    # Load test data
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    print('[info] Loading test data...')
    test_set = utils.load_test_data(test_data_zip_file=args.test_data)
    print('[info] Test data loaded.')
    print('[info] Test...')
    ckpt_evaluation(test_data=test_set, evaluate_ckptdir=args.ckptdir, bs=args.bs, logdir=args.logdir)


if __name__ == '__main__':
    main()
