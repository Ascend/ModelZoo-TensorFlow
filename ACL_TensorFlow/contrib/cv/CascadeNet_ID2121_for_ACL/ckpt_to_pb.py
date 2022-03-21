"""
to pb
"""
# coding=utf-8
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
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.pre_input import get_right_images, r2c, myPSNR, nMse
# import scipy.io as sio
import moxing as mx
from npu_bridge.npu_init import RewriterConfig
from tensorflow.python.framework import graph_util

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 1, 'Number of samples per batch')
flags.DEFINE_integer('image_size', 256, 'Image sample size in pixels')
flags.DEFINE_integer('random_seed', 0, 'Seed used to initializer rng')
flags.DEFINE_integer('num_epoch', 2000, 'number of epoch')
flags.DEFINE_integer('checkpoint_period', 10, 'save the model every time')
flags.DEFINE_integer('Dn', 11, ' the number of the convolution layers in one residual block')
flags.DEFINE_integer('Dc', 5, 'the number of the data consistency layers')
flags.DEFINE_string('model_name', 'dc', 'model name')
flags.DEFINE_string(
    'train_url', 'obs://imagenet2012-lp/cascade_log/', 'the path of train log in obs')
flags.DEFINE_string('data_url', 'obs://imagenet2012-lp/cascade_re/data/',
                    'the path of train data in obs')
flags.DEFINE_string(
    'data_test_dir', '/home/ma-user/modelarts/inputs/data_url_0/chest_test_acc3.hdf5',
    'the path of train data')
flags.DEFINE_string('last_checkpoint_dir',
                    'obs://imagenet2012-lp/cascade_log/MA-new-cascade_modelarts-11-24-11-26/output',
                    'the path of train data')
flags.DEFINE_string('last_checkpoint_dir_name',
                    '/D10-C7-24-11/', 'the path of train data')

saveDir = '/cache/saveModels'
directory = saveDir + FLAGS.last_checkpoint_dir_name
if not os.path.exists(directory):
    os.makedirs(directory)
mx.file.copy_parallel(FLAGS.last_checkpoint_dir + FLAGS.last_checkpoint_dir_name,
                      saveDir + FLAGS.last_checkpoint_dir_name)

tf.reset_default_graph()

print('Now loading the model ...')
# rec=np.empty(feature_tst.shape,dtype=np.float32)

tf.reset_default_graph()
loadChkPoint = tf.train.latest_checkpoint(directory)
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
# set precision mode allow_fp32_to_fp16  allow_mix_precision
custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes(
    'allow_fp32_to_fp16')
# # dump path
# custom_op.parameter_map['dump_path'].s = tf.compat.as_bytes(saveDir + '/')
# # set dump debug
# custom_op.parameter_map['enable_dump_debug'].b = True
# custom_op.parameter_map['dump_debug_mode'].s = tf.compat.as_bytes('all')
# custom_op.parameter_map["profiling_mode"].b = True
# custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
#     '{"output":"/cache/saveModels","task_trace":"on"}')
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # have to close
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # have to close
with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(directory + 'modelTst.meta')
    saver.restore(sess, loadChkPoint)
    graph = tf.get_default_graph()
    predT = graph.get_tensor_by_name('predTst:0')
    maskT = graph.get_tensor_by_name('mask:0')
    featureT = graph.get_tensor_by_name('feature:0')
    output_node_names = ['predTst']
    input_graph_def = graph.as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=input_graph_def,  # = sess.graph_def,
        output_node_names=output_node_names)
    with tf.gfile.GFile(saveDir + '/cascade.pb', 'wb') as fgraph:
        fgraph.write(output_graph_def.SerializeToString())

mx.file.copy_parallel('/cache/saveModels', FLAGS.train_url)
print('copy saved model to obs.')
