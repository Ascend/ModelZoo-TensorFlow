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
# Author: Tao Wu 

import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.25, 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')    # {'cora', 'citeseer', 'pubmed'}
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

flags.DEFINE_bool('save_inputs', False, 'Save input data')
flags.DEFINE_string('out_path', './saved', 'output path')
flags.DEFINE_string('ckpt_file', 'vgae_ckpt_', 'ckpt file name')
flags.DEFINE_string('pb_file', 'constant_graph_', 'pb file name')

flags.DEFINE_string('device', 'gpu', 'choose device among {cpu, gpu, npu}')
flags.DEFINE_integer('device_id', 0, 'device id for gpu')
flags.DEFINE_bool('profiling_gpu', False, 'Activate profiling GPU/CPU')
flags.DEFINE_bool('profiling_npu', False, 'Activate profiling on NPU')
flags.DEFINE_string('profiling_options', 'task_trace', 'Type of the profiling')

assert not (FLAGS.profiling_gpu and FLAGS.profiling_npu), "Cannot support profiling on both devices!"
assert (not FLAGS.profiling_gpu) or (tf.__version__[0] == '2'), "Profiling on GPU requires Tensorflow 2.x!"
