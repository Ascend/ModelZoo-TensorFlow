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

import re

import tensorflow as tf
from npu_bridge.npu_init import *
#import precision_tool.tf_config as npu_tf_config


def session(graph=None, allow_soft_placement=True,
            log_device_placement=False, allow_growth=True):
    """Return a Session with simple config."""
    # config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
    #                        log_device_placement=log_device_placement)
    # config.gpu_options.allow_growth = allow_growth

    npu_config = tf.ConfigProto(allow_soft_placement=allow_soft_placement, log_device_placement=log_device_placement)
    custom_op = npu_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["enable_data_pre_proc"].b = True  # Ԥ������


    #custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
    #custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    npu_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # ������ʽ�ر�remap
    npu_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    #npu_config = npu_tf_config.session_dump_config(npu_config, action='overflow')
    #npu_config = npu_tf_config.session_dump_config(npu_config, action='fusion_switch')
    #npu_config = npu_tf_config.session_dump_config(npu_config, action='')
    #npu_config = npu_tf_config.session_dump_config(npu_config, action='dump')

    return tf.Session(config=npu_config, graph=graph)


def print_tensor(tensors):
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]

    for i, tensor in enumerate(tensors):
        ctype = str(type(tensor))
        if 'Tensor' in ctype:
            type_name = 'Tensor'
        elif 'Variable' in ctype:
            type_name = 'Variable'
        else:
            raise Exception('Not a Tensor or Variable!')

        print(str(i) + (': %s("%s", shape=%s, dtype=%s, device=%s)'
                        % (type_name, tensor.name, str(tensor.get_shape()),
                           tensor.dtype.name, tensor.device)))

prt = print_tensor


def shape(tensor):
    sp = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in sp]


def summary(tensor_collection,
            summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram'],
            scope=None):
    """Summary.

    usage:
        1. summary(tensor)
        2. summary([tensor_a, tensor_b])
        3. summary({tensor_a: 'a', tensor_b: 'b})
    """
    def _summary(tensor, name, summary_type):
        """Attach a lot of summaries to a Tensor."""
        if name is None:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            name = re.sub('%s_[0-9]*/' % 'tower', '', tensor.name)
            name = re.sub(':', '-', name)

        summaries = []
        if len(tensor.shape) == 0:
            summaries.append(tf.summary.scalar(name, tensor))
        else:
            if 'mean' in summary_type:
                mean = tf.reduce_mean(tensor)
                summaries.append(tf.summary.scalar(name + '/mean', mean))
            if 'stddev' in summary_type:
                mean = tf.reduce_mean(tensor)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
                summaries.append(tf.summary.scalar(name + '/stddev', stddev))
            if 'max' in summary_type:
                summaries.append(tf.summary.scalar(name + '/max', tf.reduce_max(tensor)))
            if 'min' in summary_type:
                summaries.append(tf.summary.scalar(name + '/min', tf.reduce_min(tensor)))
            if 'sparsity' in summary_type:
                summaries.append(tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor)))
            if 'histogram' in summary_type:
                summaries.append(tf.summary.histogram(name, tensor))
        return tf.summary.merge(summaries)

    if not isinstance(tensor_collection, (list, tuple, dict)):
        tensor_collection = [tensor_collection]

    with tf.name_scope(scope, 'summary'):
        summaries = []
        if isinstance(tensor_collection, (list, tuple)):
            for tensor in tensor_collection:
                summaries.append(_summary(tensor, None, summary_type))
        else:
            for tensor, name in tensor_collection.items():
                summaries.append(_summary(tensor, name, summary_type))
        return tf.summary.merge(summaries)


def counter(start=0, scope=None):
    with tf.variable_scope(scope, 'counter'):
        counter = tf.get_variable(name='counter',
                                  initializer=tf.constant_initializer(start),
                                  shape=(),
                                  dtype=tf.int64)
        update_cnt = tf.assign(counter, tf.add(counter, 1))
        return counter, update_cnt
