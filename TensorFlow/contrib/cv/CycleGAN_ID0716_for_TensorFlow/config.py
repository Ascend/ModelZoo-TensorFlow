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


import tensorflow as tf
import os

"""
Here, define the configuration of tensorflow session
For different chips, the config is not the same. 
"""


def make_config(FLAGS):
    chip = FLAGS.chip.lower()
    tf.logging.info("chip is [%s]", chip)

    if chip == 'cpu':
        config = tf.ConfigProto()
    elif chip == 'gpu':
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
    elif chip == 'npu':
        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        # Performance Profiling
        # refer to link:https://support.huaweicloud.com/Development-tg-cann202training1/atlasprofilingtrain_16_0003.html
        if FLAGS.profiling:
            # work_dir = os.getcwd()
            # profiling_dir = os.path.join(work_dir, "npu_profiling")
            profiling_dir = os.path.join(FLAGS.result, 'npu_profiling')
            if not os.path.exists(profiling_dir):
                os.makedirs(profiling_dir)

            options = '{"output": "%s", \
                        "task_trace": "on", \
                        "aicpu": "on"}' % (profiling_dir)

            custom_op.parameter_map["profiling_mode"].b = True
            custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(options)

    else:
        raise RuntimeError('chip [%s] has not supported' % chip)

    return config
