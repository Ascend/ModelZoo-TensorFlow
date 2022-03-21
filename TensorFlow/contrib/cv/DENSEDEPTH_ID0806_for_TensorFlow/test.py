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
import glob
import argparse
import matplotlib
from npu_bridge.npu_init import *

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import predict, load_images, display_images
from matplotlib import pyplot as plt
import tensorflow as tf
from model import create_model
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["enable_data_pre_proc"].b = True
custom_op.parameter_map["graph_run_mode"].i = 0
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")
# custom_op.parameter_map["op_select_implmode"].s = tf.compat.as_bytes("high_precision")
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

npu_keras_sess = set_keras_session_npu_config(config=sess_config)

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--ckptdir', default='/data1/NRE_Check/wx1056345/ID0806_densedepth/result/models/1631771923-n85-e15-bs2-lr0.0001-densedepth_nyu/ckpt_npu/', type=str, help='Trained ckpt model file.')
parser.add_argument('--input', default='./examples/*.png', type=str, help='Input filename or folder.')
parser.add_argument('--output', default='./result/test/test.png', type=str, help='output image filename.')
args = parser.parse_args()

print('Loading model...')

# Load model into NPU
model = create_model()
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(
    args.ckptdir)).assert_existing_objects_matched().run_restore_ops()

print('\nModel loaded ({0}).'.format(args.ckptdir))

# Input images
inputs = load_images(glob.glob(args.input), resolution=480)

print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)

# matplotlib problem on ubuntu terminal fix
# matplotlib.use('TkAgg')

# Display results

save_path = args.output
if not os.path.exists(save_path):
    os.makedirs(os.path.dirname(save_path))
print("output path:{}".format(save_path))

viz = display_images(outputs.copy(), inputs.copy())
plt.figure(figsize=(10, 5))
plt.imshow(viz)
plt.savefig(save_path)
plt.show()


close_session(npu_keras_sess)