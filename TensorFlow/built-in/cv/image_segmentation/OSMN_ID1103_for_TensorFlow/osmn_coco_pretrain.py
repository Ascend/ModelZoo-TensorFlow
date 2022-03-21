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
"""
One-Shot Modulation Netowrk
Training on MS-COCO
"""
from npu_bridge.npu_init import *
import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import argparse
import osmn
from dataset_coco import Dataset
import common_args
def add_arguments(parser):
    group = parser.add_argument_group('Additional params')
    group.add_argument(
            '--data_path',
            type=str,
            required=False,
            default='/raid/ljyang/data/MS_COCO',
            help='Path to the MS COCO dataset')
    group.add_argument(
            '--vis_mod_model_path',
            type=str,
            required=False,
            default='models/vgg_16.ckpt',
            help='Model to initialize visual modulator')
    group.add_argument(
            '--seg_model_path',
            type=str,
            required=False,
            default='models/vgg_16.ckpt',
            help='Model to initialize segmentation model')

    group.add_argument(
            '--whole_model_path',
            type=str,
            required=False,
            default='',
            help='Model to initialize the whole model')
    group.add_argument(
            '--im_size',
            nargs=2, type=int,
            required = False,
            default=[400, 400],
            help='Input image size')
    group.add_argument(
            '--data_aug_scales',
            nargs='+', type=float,
            required=False,
            default=[0.8, 1, 1.2],
            help='Image scales for data augmentation')

print(" ".join(sys.argv[:]))

parser = argparse.ArgumentParser()
common_args.add_arguments(parser)
add_arguments(parser)
args = parser.parse_args()
baseDir = args.data_path
# User defined parameters
train_path = os.path.join(baseDir, 'train2017/{:012d}.jpg')
val_path = os.path.join(baseDir, 'val2017/{:012d}.jpg')
train_file = os.path.join(baseDir, 'annotations/instances_train2017.json')
val_file = os.path.join(baseDir, 'annotations/instances_val2017.json')
print(args)
sys.stdout.flush()
# Define Dataset
dataset = Dataset(train_file, val_file, train_path, val_path, args, data_aug=True)
# Train parameters
logs_path = args.model_save_path
max_training_iters = args.training_iters
learning_rate = args.learning_rate
save_step = args.save_iters
display_step = args.display_iters
batch_size = args.batch_size
resume_training = args.resume_training
use_image_summary = args.use_image_summary
## default config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
custom_op.parameter_map["dynamic_input"].b = 1
custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

#custom_op.parameter_map["mix_compile_mode"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
if not args.only_testing:
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            osmn.train_finetune(dataset, args, learning_rate, logs_path, max_training_iters,
                                 save_step, display_step, global_step, batch_size = batch_size, config=config, 
                                 iter_mean_grad=1, use_image_summary=use_image_summary, resume_training=resume_training, ckpt_name='osmn')


