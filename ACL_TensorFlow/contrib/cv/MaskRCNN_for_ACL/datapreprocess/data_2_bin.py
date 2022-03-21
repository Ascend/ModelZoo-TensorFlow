# Copyright 2020 Huawei Technologies Co., Ltd
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

from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
import sys
import os
import six

import dataloader
from hyperparameters import flags_to_params
from hyperparameters import params_dict
from configs import mask_rcnn_config


flags.DEFINE_string(
    'distribution_strategy',
    default='multi_worker_gpu',
    help='Distribution strategy or estimator type to use. One of'
    '"multi_worker_gpu"|"tpu".')

# Parameters for MultiWorkerMirroredStrategy
flags.DEFINE_string(
    'worker_hosts',
    default=None,
    help='Comma-separated list of worker ip:port pairs for running '
    'multi-worker models with distribution strategy.  The user would '
    'start the program on each host with identical value for this flag.')
flags.DEFINE_integer(
    'task_index', 0, 'If multi-worker training, the task_index of this worker.')
flags.DEFINE_integer(
    'num_gpus',
    default=1,
    help='Number of gpus when using collective all reduce strategy.')
flags.DEFINE_integer(
    'worker_replicas',
    default=0,
    help='Number of workers when using collective all reduce strategy.')

# TPUEstimator parameters
flags.DEFINE_integer(
    'num_cores', default=None, help='Number of TPU cores for training')
flags.DEFINE_multi_integer(
    'input_partition_dims', None,
    'A list that describes the partition dims for all the tensors.')
flags.DEFINE_bool(
    'transpose_input',
    default=None,
    help='Use TPU double transpose optimization')
flags.DEFINE_string(
    'tpu_job_name', None,
    'Name of TPU worker binary. Only necessary if job name is changed from'
    ' default tpu_worker.')

# Model specific paramenters
flags.DEFINE_string('mode', 'train',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')

flags.DEFINE_string(
    'training_file_pattern', '/root/33334/data_reMake_includeMask/train-000*',
    ' ')

flags.DEFINE_string(
    'resnet_checkpoint', '/root/33334/resnet34/model.ckpt-28152',
    ' ')

flags.DEFINE_string(
    'validation_file_pattern', '/root/33334/fast-rcnn/data_new/coco_official_2017/tfrecord/val*',
    ' ')
flags.DEFINE_string(
    'binfilepath', './data',
    ' ')


# modelarts
flags.DEFINE_string(
    'data_dir', None,
    'path to dataset.')
flags.DEFINE_string(
    'model_dir', None,
    'model_dir')
flags.DEFINE_string(
    'tpu', None,
    'tpu')    


FLAGS = flags.FLAGS
FLAGS(sys.argv)

def data2bin():
    params = params_dict.ParamsDict(
        mask_rcnn_config.MASK_RCNN_CFG, mask_rcnn_config.MASK_RCNN_RESTRICTIONS)
    params = params_dict.override_params_dict(
        params, None, is_strict=True)
    params = params_dict.override_params_dict(
        params, None, is_strict=True)
    params = flags_to_params.override_params_from_input_flags(params, FLAGS)
    params.validate()
    params.lock()

    print("params：", params)
    print("validation_file_pattern：", params.validation_file_pattern)
    os.system("mkdir {}".format(FLAGS.binfilepath))

    graph = tf.Graph()
    with graph.as_default():
        newparam = {}
        newparam['validation_file_pattern'] = params.validation_file_pattern
        newparam['resize_method'] = params.resize_method
        newparam['image_size'] =  params.image_size
        newparam['max_level'] = params.max_level
        newparam['short_side'] = params.short_side
        newparam['long_side'] =params.long_side
        newparam['max_level'] =params.max_level
        newparam['precision'] =params.precision
        newparam['include_groundtruth_in_features'] =params.include_groundtruth_in_features
        newparam['visualize_images_summary']= params.visualize_images_summary
        newparam['include_mask']= params.include_mask
        newparam['use_category'] = params.use_category
        newparam['skip_crowd_during_training'] = params.skip_crowd_during_training
        newparam['input_rand_hflip'] = params.input_rand_hflip
        newparam['resize_method'] = params.resize_method
        newparam['aug_scale_min'] = params.aug_scale_min
        newparam['aug_scale_max'] = params.aug_scale_max
        newparam['gt_mask_size'] = params.gt_mask_size
        newparam['short_side'] = params.short_side
        newparam['long_side'] = params.long_side
        newparam['num_scales'] = params.num_scales
        newparam['aspect_ratios'] = params.aspect_ratios
        newparam['anchor_scale'] = params.anchor_scale
        newparam['num_classes'] = params.num_classes
        newparam['rpn_positive_overlap'] = params.rpn_positive_overlap
        newparam['rpn_negative_overlap'] = params.rpn_negative_overlap
        newparam['rpn_batch_size_per_im'] = params.rpn_batch_size_per_im
        newparam['rpn_fg_fraction'] = params.rpn_fg_fraction
        newparam['precision'] = params.precision
        newparam['transpose_input'] = params.transpose_input
        newparam['backbone'] = params.backbone
        newparam['conv0_space_to_depth_block_size'] = params.conv0_space_to_depth_block_size

        eval_input_fn = dataloader.InputReader(
            params.validation_file_pattern,
            mode=tf.estimator.ModeKeys.PREDICT,
            num_examples=params.eval_samples,
            use_instance_mask=params.include_mask)
        dataset = eval_input_fn(newparam)

        iterator = dataset.make_initializable_iterator()
        features = iterator.get_next()

        images = features['features']['images']
        image_info = features['features']['image_info']
        source_ids = features['features']['source_ids']
        images = tf.cast(images, tf.float32)
        image_info = tf.cast(image_info, tf.float32)
        source_ids = tf.cast(source_ids, tf.float32)

    with tf.Session(graph=graph) as sess:
        sess.run(iterator.initializer)

        a = 0
        while (a < 5000):
            try:
                ouput = sess.run([images,image_info,source_ids])
                ouput[0].tofile(os.path.join("{}".format(FLAGS.binfilepath), "{}_images.bin".format(a)))
                ouput[1].tofile(os.path.join("{}".format(FLAGS.binfilepath), "{}_image_info.bin".format(a)))
                ouput[2].tofile(os.path.join("{}".format(FLAGS.binfilepath), "{}_source_ids.bin".format(a)))
                
                a = a + 1
            except:
                print('iteraotr is null~~~~~')
                print('~~~~a', a)
                break

if __name__ == '__main__':
    data2bin()


