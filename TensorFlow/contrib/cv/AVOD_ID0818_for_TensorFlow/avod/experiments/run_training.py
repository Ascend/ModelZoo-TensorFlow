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


"""Detection model trainer.

This runs the DetectionModel trainer.
"""
from npu_bridge.npu_init import *

import argparse
import os

import tensorflow as tf

import avod
import avod.builders.config_builder_util as config_builder
from avod.builders.dataset_builder import DatasetBuilder
from avod.core.models.avod_model import AvodModel
from avod.core.models.rpn_model import RpnModel
from avod.core import trainer

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def train(model_config, train_config, dataset_config, args):

    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 use_defaults=False)

    train_val_test = 'train'
    model_name = model_config.model_name

    with tf.Graph().as_default():
        if model_name == 'rpn_model':
            model = RpnModel(model_config,
                             train_val_test=train_val_test,
                             dataset=dataset)
        elif model_name == 'avod_model':
            model = AvodModel(model_config,
                              train_val_test=train_val_test,
                              dataset=dataset)
        else:
            raise ValueError('Invalid model_name')
        
        trainer.train(model, train_config, args)


def main(_):
    parser = argparse.ArgumentParser()

    # Defaults
    default_pipeline_config_path = avod.root_dir() + \
        '/configs/avod_cars_example.config'
    default_data_split = 'train'
    default_device = '0'

    parser.add_argument('--pipeline_config',
                        type=str,
                        dest='pipeline_config_path',
                        default=default_pipeline_config_path,
                        help='Path to the pipeline config')

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        default=default_data_split,
                        help='Data split for training')

    parser.add_argument('--train_steps',
                        type=int,
                        default=200000,
                        help='train_steps default 200000')

    parser.add_argument('--checkpoint_interval',
                        type=int,
                        default=1000,
                        help='checkpoint_interval default 1000')

    parser.add_argument('--summary_interval',
                        type=int,
                        default=10,
                        help='summary_interval default 1000')

    parser.add_argument('--initial_learning_rate',
                        type=float,
                        default=0.0001,
                        help='initial_learning_rate default 0.0001')
                        
    parser.add_argument('--precision_mode',
                        type=str,
                        default="allow_mix_precision",
                        help='precision_mode default allow_mix_precision; allow_fp32_to_fp16 | force_fp16 | must_keep_origin_dtype | allow_mix_precision')
    
    parser.add_argument('--profiling',
                        type=int,
                        default=0,
                        help='profiling default 0 (False)')

    parser.add_argument('--profiling_dump_path',
                        type=str,
                        default="./output/profiling",
                        help='profiling_dump_path default: output/profiling')                      
                        

    args = parser.parse_args()

    # Parse pipeline config
    model_config, train_config, _, dataset_config = \
        config_builder.get_configs_from_pipeline_file(
            args.pipeline_config_path, is_training=True)
    
    train_config.max_iterations = args.train_steps
    train_config.checkpoint_interval = args.checkpoint_interval
    train_config.summary_interval = args.summary_interval
    train_config.optimizer.adam_optimizer.learning_rate.exponential_decay_learning_rate.initial_learning_rate = args.initial_learning_rate

    # Overwrite data split
    dataset_config.data_split = args.data_split

    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = default_device

    train(model_config, train_config, dataset_config, args)


if __name__ == '__main__':
    tf.app.run()

