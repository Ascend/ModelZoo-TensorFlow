# Copyright 2022 Huawei Technologies Co., Ltd
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

import tensorflow as tf


def _npu_config(mix_precision, is_distributed):
    """Prepare NPU tf.Session config
    
    Args:
        mix_precision: boolean, whether to use mix precision. 
        is_distributed: boolean, whether in distributed scenario.
        
    Returns:
        A tf.ConfigProto instance.
    """
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["enable_data_pre_proc"].b = False
    custom_op.parameter_map["mix_compile_mode"].b = False
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["graph_memory_max_size"].s = \
        tf.compat.as_bytes(str(28*1024 * 1024 * 1024))
    custom_op.parameter_map["variable_memory_max_size"].s = \
        tf.compat.as_bytes(str(3*1024 * 1024 * 1024))
    
    if mix_precision:
        custom_op.parameter_map["precision_mode"].s = \
            tf.compat.as_bytes("allow_mix_precision")
    if is_distributed:     
        config.graph_options.rewrite_options.optimizers.extend(
            ["pruning",
             "function",
             "constfold",
             "shape",
             "arithmetic",
             "loop",
             "dependency",
             "layout",
             "memory",
             "GradFusionOptimizer"])

    from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    return config


def _cpu_config():
    """Prepare CPU tf.Session config

    Returns:
        A tf.ConfigProto instance.
    """
    return tf.ConfigProto()


def get_sess_config(device='npu', xla=False, mix_precision=True, is_distributed=False):
    """Build session config.

    Args:
        device: str, what type of hardware to use.
        xla: boolean, whether to use xla.
        mix_precision: boolean, whether to use mix precision.
        is_distributed: boolean, whether in distributed scenario.

    Returns:
        A tf.ConfigProto instance.
    """
    if device == 'npu':
        return _npu_config(mix_precision, is_distributed)
    elif device == 'cpu':
        return _cpu_config()
    else:
        raise KeyError('Unsupported device: {}'.format(device))
