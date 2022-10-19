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

def make_config(FLAGS):
    '''
    FLAGS:input args
    '''
    chip = FLAGS.chip.lower()
    tf.logging.info("chip is [%s]", chip)

    if chip == 'cpu':
        config = tf.ConfigProto()
    elif chip == 'gpu':
        ## GPU config
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
    elif chip == 'npu':
        import npu_bridge
        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["hcom_parallel"].b = True
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

        ## Auto Tune
        ## refer to link:https://support.huaweicloud.com/TensorFlowdevg-cann202training1/atlasmprtg_13_0031.html
        if FLAGS.npu_auto_tune:
            print("===>>>Using npu auto tune tool")
            custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
            work_dir = os.getcwd()
            auto_tune_dir = os.path.join(work_dir, "npu_auto_tune")
            os.environ["TUNE_BANK_PATH"] = auto_tune_dir
            if not os.path.exists(auto_tune_dir):
                os.makedirs(auto_tune_dir)

        ## Dump data for comparing 
        ## refer to link: https://support.huaweicloud.com/Development-tg-cann202training1/atlasacctrain_16_0004.html
        if FLAGS.npu_dump_data:
            work_dir = os.getcwd()
            dump_data_dir = os.path.join(work_dir, "npu_dump_data")
            if not os.path.exists(dump_data_dir):
                os.makedirs(dump_data_dir)

            custom_op.parameter_map["enable_dump"].b = True
            custom_op.parameter_map['dump_path'].s = tf.compat.as_bytes(dump_data_dir)
            custom_op.parameter_map['dump_step'].s = tf.compat.as_bytes("0")
            custom_op.parameter_map['dump_mode'].s = tf.compat.as_bytes("all")

        ## Dump Net Computation Graph
        ## refer to link:https://support.huaweicloud.com/tensorflowdevg-cann330alphaXtraining/atlasmprtg_13_0077.html
        if FLAGS.npu_dump_graph:
            os.environ["DUMP_GE_GRAPH"] = "2"
            work_dir = os.getcwd()
            dump_graph_dir = os.path.join(work_dir, "npu_dump_graph")
            if not os.path.exists(dump_graph_dir):
                os.makedirs(dump_graph_dir)
            os.environ["DUMP_GRAPH_PATH"] = dump_graph_dir

        ## Performance Profiling
        ## refer to link:https://support.huaweicloud.com/Development-tg-cann202training1/atlasprofilingtrain_16_0003.html
        if FLAGS.npu_profiling:
            work_dir = os.getcwd()
            profiling_dir = os.path.join(work_dir, "npu_profiling")
            if not os.path.exists(profiling_dir):
                os.makedirs(profiling_dir)
            
            ## For different Net, the fp and bp node maybe different, please change them.
            fp_node_name = "conv1/Conv2D"
            bp_node_name = "gradients/conv1/Conv2D_grad/Conv2DBackpropFilter"
            options = '{"output": "%s", \
                        "training_trace": "on", \
                        "task_trace": "on", \
                        "aicpu": "on", \
                        "fp_point": "%s",\
                        "bp_point": "%s"}' % (profiling_dir, fp_node_name, bp_node_name)

            custom_op.parameter_map["profiling_mode"].b = True
            custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(options)
    
    else:
        raise RuntimeError('chip [%s] has not supported' % chip)

    return config