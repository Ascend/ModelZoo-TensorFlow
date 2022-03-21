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
from npu_bridge.npu_init import *
import os
import argparse
import tensorflow as tf
from network import GraphNet


#def configure():
# training
flags = tf.app.flags
flags.DEFINE_integer('max_step', 10000, '# of step for training')
flags.DEFINE_integer('summary_interval', 10, '# of step to save summary')
flags.DEFINE_float('learning_rate', 0.1, 'learning rate')
flags.DEFINE_boolean('is_train', True, 'is train')
flags.DEFINE_integer('class_num', 7, 'output class number')
# Debug
flags.DEFINE_string('logdir', './logdir', 'Log dir')
flags.DEFINE_string('modeldir', './modeldir', 'Model dir')
flags.DEFINE_string('model_name', 'model', 'Model file name')
flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
flags.DEFINE_integer('test_step', 0, 'Test or predict model at this step')
# network architecture
flags.DEFINE_integer('ch_num', 8, 'channel number')
flags.DEFINE_integer('layer_num', 2, 'block number')
flags.DEFINE_float('adj_keep_r', 0.999, 'dropout keep rate')
flags.DEFINE_float('keep_r', 0.16, 'dropout keep rate')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('k', 8, 'top k')
flags.DEFINE_string('first_conv', 'simple_conv', 'simple_conv, chan_conv')
flags.DEFINE_string('second_conv', 'graph_conv', 'graph_conv, simple_conv')
flags.DEFINE_boolean('use_batch', True, 'use batch training')
flags.DEFINE_integer('batch_size', 2500, 'batch size number')
flags.DEFINE_integer('center_num', 1500, 'start center number')
# fix bug of flags
flags.FLAGS.__dict__['__parsed'] = False
    #return flags.FLAGS


def main(_):
    conf = flags.FLAGS
    config = tf.ConfigProto()
    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    GraphNet(tf.Session(config=config), conf).train()


if __name__ == '__main__':
    # configure which gpu or cpu to use
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run()

