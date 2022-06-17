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
import tensorflow as tf
import json
import os
import argparse
import sys
from src.model import model_fn
from src.input_pipeline import Pipeline



current_path = sys.path[0]
tf.logging.set_verbosity('INFO')
dir_path = os.path.dirname(os.path.abspath(__file__))
#CONFIG = current_path+'/config.json'
CONFIG = os.path.join(current_path,'config.json')
GPU_TO_USE = '0'


params = json.load(open(CONFIG))
# params = json.load(open(r'./config.json'))
model_params = params['model_params']
input_params = params['input_pipeline_params']
parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, default=input_params['num_steps'])
parser.add_argument("--data_path", type=str, help="add data_file")
parser.add_argument("--output_path", type=str, help="add output_file")
parser.add_argument("--log_step_count_steps", type=int, default=10 ,help="add output_file")
iconfig = parser.parse_args()
train_datafile=iconfig.data_path+"train_shards/"
val_datafile=iconfig.data_path+"val_shards/"
output_infofile=iconfig.output_path

def get_input_fn_(is_training=True):

    image_size = input_params['image_size'] #if is_training else None
    # (for evaluation i use images of different sizes)
    dataset_path = train_datafile if is_training else val_datafile
    batch_size = input_params['batch_size'] if is_training else 1
    # for evaluation it's important to set batch_size to 1

    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        with tf.name_scope('input_pipeline'):
            pipeline = Pipeline(
                filenames,
                batch_size=batch_size, image_size=image_size,
                repeat=is_training, shuffle=is_training,
                augmentation=is_training
            )
            features, labels = pipeline.get_batch()
        return features, labels
    features, labels=input_fn()
    return features, labels



config_proto = tf.ConfigProto()
custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config_proto.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
custom_op.parameter_map["dynamic_input"].b = True
custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("dynamic_execute")
custom_op.parameter_map["dynamic_inputs_shape_range"].s = tf.compat.as_bytes("getnext:[16,1024,1024,3],[16,1~2048,4],[16]")
npu_config=NPURunConfig(
    model_dir=output_infofile,
    save_checkpoints_steps=5000,
    log_step_count_steps=1,
    session_config=config_proto,
)


estimator = tf.estimator.Estimator(model_fn, params=model_params, config=npu_config)
estimator.train(input_fn=lambda :get_input_fn_(is_training=True),max_steps=iconfig.step, hooks=npu_hooks_append())