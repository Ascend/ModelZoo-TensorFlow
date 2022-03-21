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
import os
from pathlib import Path

import tensorflow as tf
from npu_bridge.npu_init import RewriterConfig

PLATFORM = 'ascend'
WORKSPACE_DIR = os.environ['PRJPATH']
SOURCE_DIR = os.path.join(WORKSPACE_DIR, 'src')
CHECKPOINT_DIR = os.path.join(WORKSPACE_DIR, 'checkpoint')
LOG_DIR = os.path.join(WORKSPACE_DIR, 'log')

# Device
ALLOW_MULTI_DEVICE = False
if os.environ.get('IS_MODELARTS', 'False') != 'True' and 'ASCEND_DEVICE_ID' not in os.environ:
    os.environ['ASCEND_DEVICE_ID'] = '4'

# Auto Tune (ascend)
AUTO_TUNE = False
if AUTO_TUNE:
    os.environ['TE_PARALLEL_COMPILER'] = '32'

# Log (ascend)
PRINT_HOST_LOG = True
DUMP = False
DUMP_DIR = '/cache/dump_data'
PROFILING = False
PROFILING_DIR = '/home/test_user02/shawn/profiling'

if PRINT_HOST_LOG:
    os.environ['SLOG_PRINT_TO_STDOUT'] = "1"


# Session Config
def get_sess_cfg(sess_cfg=None):
    """get session config

    Args:
        sess_cfg ([sess_cfg], optional): session config. Defaults to None.
    Returns:
        sess_cfg
    """
    if sess_cfg is None:
        sess_cfg = tf.ConfigProto()
    custom_optimizer = sess_cfg.graph_options.rewrite_options.custom_optimizers.add()
    custom_optimizer.name = "NpuOptimizer"
    custom_optimizer.parameter_map['use_off_line'].b = True
    # custom_optimizer.parameter_map['enable_data_pre_proc'].b = True
    if AUTO_TUNE:
        custom_optimizer.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
    # custom_optimizer.parameter_map["mix_compile_mode"].b = True
    # custom_optimizer.parameter_map['precision_mode'].s = tf.compat.as_bytes("allow_mix_precision")
    if DUMP:
        custom_optimizer.parameter_map["enable_dump"].b = True
        custom_optimizer.parameter_map["dump_path"].s = tf.compat.as_bytes(DUMP_DIR)
        custom_optimizer.parameter_map["dump_step"].s = tf.compat.as_bytes("0|5|10")
        custom_optimizer.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
    if PROFILING:
        custom_optimizer.parameter_map["profiling_mode"].b = True
        custom_optimizer.parameter_map["profiling_options"].s = tf.compat.as_bytes(
            '{"output":"%s","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"%s","bp_point":"%s"}' %
            (PROFILING_DIR, 'backbone/conv2d/Conv2D',
             'train_op/gradients/backbone/conv2d/Conv2D_grad/Conv2DBackpropFilter'))
    sess_cfg.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess_cfg.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    return sess_cfg
