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
from npu_bridge.npu_init import *
import time
import tensorflow as tf
from network import PixelDCN

"""
This file provides configuration to build U-NET for semantic segmentation.

"""

# training
flags = tf.app.flags
flags.DEFINE_string('option', 'train', 'actions: train, test, or predict')
flags.DEFINE_string('precision_mode', 'allow_mix_precision', 'precision mode')
flags.DEFINE_string('over_dump', 'False', 'if or not over detection')
flags.DEFINE_string('over_dump_path', './overdump', 'over dump path')
flags.DEFINE_string('data_dump_flag', 'False', 'data dump flag')
flags.DEFINE_string('data_dump_step', '10', 'data dump step')
flags.DEFINE_string('data_dump_path', './datadump', 'data dump path')
flags.DEFINE_string('profiling', '.False', 'if or not profiling for performance debug')
flags.DEFINE_string('profiling_dump_path', './profiling', 'profiling path')
flags.DEFINE_string('loss_scale', 'True', 'enable loss scale ,default is True')
flags.DEFINE_string('autotune', 'False', 'whether to enable autotune, default is False')


flags.DEFINE_integer('max_step', 6, '# of step for training')
flags.DEFINE_integer('test_interval', 100, '# of interval to test a model')
flags.DEFINE_integer('save_interval', 2, '# of interval to save  model')
flags.DEFINE_integer('summary_interval', 100, '# of step to save summary')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
# data
flags.DEFINE_string('data_path', './dataset/', 'Name of data directory')
flags.DEFINE_string('train_data', 'training3d.h5', 'Training data')
flags.DEFINE_string('valid_data', 'validation3d.h5', 'Validation data')
flags.DEFINE_string('test_data', 'testing3d.h5', 'Testing data')
flags.DEFINE_string('data_type', '3D', '2D data or 3D data')
flags.DEFINE_integer('batch', 2, 'batch size')
flags.DEFINE_integer('channel', 1, 'channel size')
flags.DEFINE_integer('depth', 16, 'depth size')
flags.DEFINE_integer('height', 256, 'height size')
flags.DEFINE_integer('width', 256, 'width size')
# Debug
flags.DEFINE_string('logdir', './logdir', 'Log dir')
flags.DEFINE_string('modeldir', './modeldir', 'Model dir')
flags.DEFINE_string('sampledir', './samples/', 'Sample directory')
flags.DEFINE_string('model_name', 'model', 'Model file name')
flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
flags.DEFINE_integer('test_step', 0, 'Test or predict model at this step')
flags.DEFINE_integer('random_seed', int(time.time()), 'random seed')
# network architecture
flags.DEFINE_integer('network_depth', 5, 'network depth for U-Net')
flags.DEFINE_integer('class_num', 2, 'output class number')
flags.DEFINE_integer('start_channel_num', 16,
                     'start number of outputs for the first conv layer')
flags.DEFINE_string(
    'conv_name', 'conv',
    'Use which conv op in decoder: conv or ipixel_cl')
flags.DEFINE_string(
    'deconv_name', 'ipixel_dcl',
    'Use which deconv op in decoder: deconv, pixel_dcl, ipixel_dcl')
flags.DEFINE_string(
    'action', 'concat',
    'Use how to combine feature maps in pixel_dcl and ipixel_dcl: concat or add')
# fix bug of flags
flags.FLAGS.__dict__['__parsed'] = False
FLAGS = tf.app.flags.FLAGS

def main(_):
    if FLAGS.option not in ['train', 'test', 'predict']:
        print('invalid option: ', FLAGS.option)
        print("Please input a option: train, test, or predict")
    else:
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(FLAGS.precision_mode)
        if FLAGS.data_dump_flag.strip() == "True":
            custom_op.parameter_map["enable_dump"].b = True
            custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(FLAGS.data_dump_path)
            custom_op.parameter_map["dump_step"].s = tf.compat.as_bytes(FLAGS.data_dump_step)
            custom_op.parameter_map["dump_mode"].s = tf.compat.as_bytes("all")
        if FLAGS.over_dump.strip() == "True":
            # dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限
            custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes(FLAGS.over_dump_path)
            # enable_dump_debug：是否开启溢出检测功能
            custom_op.parameter_map["enable_dump_debug"].b = True
            # dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
            custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
        if FLAGS.profiling.strip() == "True":
            custom_op.parameter_map["profiling_mode"].b = False
            profilingvalue = (
                    '{"output":"%s","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":""}' % (
                FLAGS.profiling_dump_path))
            custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(profilingvalue)
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        model = PixelDCN(tf.Session(config=config), FLAGS)
        getattr(model, FLAGS.option)()

if __name__ == '__main__':
    # configure which gpu or cpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()
