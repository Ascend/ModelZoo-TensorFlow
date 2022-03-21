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
from src.model import model_fn
from src.input_pipeline import Pipeline


current_path = os.path.dirname(__file__)
tf.logging.set_verbosity('INFO')
dir_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = current_path+'/config.json'
GPU_TO_USE = '0'


params = json.load(open(CONFIG))
model_params = params['model_params']
input_params = params['input_pipeline_params']
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="add data_file")
parser.add_argument("--output_path", type=str, help="add output_file")
iconfig = parser.parse_args()
train_datafile=iconfig.data_path+"train_shards/"
val_datafile=iconfig.data_path+"val_shards/"
output_infofile=iconfig.output_path
print(train_datafile)
print(output_infofile)


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
        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(
                filenames,
                batch_size=batch_size, image_size=image_size,
                repeat=is_training, shuffle=is_training,
                augmentation=is_training
            )
            features, labels = pipeline.get_batch()
            features['images'] = util.set_graph_exec_config(features['images'], dynamic_input=True, dynamic_graph_execute_mode="dynamic_execute", dynamic_inputs_shape_range='data:[1~16,1024,1024,3],[1~16,0~2048,4],[1~16]')

        return features, labels
    features, labels=input_fn()
    return features, labels

config = tf.ConfigProto()
config.gpu_options.visible_device_list = GPU_TO_USE
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=output_infofile,
    session_config=config,
    save_summary_steps=5000,
    save_checkpoints_steps=5000,
    log_step_count_steps=10
)


estimator = tf.estimator.Estimator(model_fn, params=model_params, config=npu_run_config_init(run_config=run_config))
estimator.evaluate(input_fn=lambda :get_input_fn_(is_training=False),  hooks=npu_hooks_append())