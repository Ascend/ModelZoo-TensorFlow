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
"""
Here, define the configuration of tensorflow session
For different chips, the config is not the same. 
"""
import tensorflow as tf
import os
from npu_bridge.npu_init import *

def make_config(chip):
    #chip = FLAGS.chip.lower()
    #tf.logging.info("chip is [%s]", chip)

    if chip == 'cpu':
        config = tf.ConfigProto()
    elif chip == 'gpu':
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
    elif chip == 'npu':
        # from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["mix_compile_mode"].b = True
        custom_op.parameter_map["use_off_line"].b = True
        # change
        custom_op.parameter_map["hcom_parallel"].b = True
        
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["customize_dtypes"].s = tf.compat.as_bytes("./switch_config.txt")

        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    else:
        raise RuntimeError('chip [%s] has not supported' % chip)

    return config
