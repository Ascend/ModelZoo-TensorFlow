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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import argparse
import tensorflow as tf
from official1.utils1.arg_parsers import parsers
from official1.utils1.logging import hooks_helper
from DAN_V2 import get_pts
import numpy as np
import cv2
import os
import tests
import time


def validate_batch_size_for_multi_gpu(batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of
    available GPUs.

    Note that this should eventually be handled by replicate_model_fn
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.
    """
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
    if not num_gpus:
        raise ValueError('Multi-GPU mode was specified, but no GPUs '
                         'were found. To use CPU, run without --multi_gpu.')

    remainder = batch_size % num_gpus
    if remainder:
        err = ('When running with multiple GPUs, batch size '
               'must be a multiple of the number of available GPUs. '
               'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
               ).format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)


def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_parallel_calls=1,
                           examples_per_epoch=0, multi_gpu=False):
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.repeat(num_epochs)

    if multi_gpu:
        total_examples = num_epochs * examples_per_epoch
        dataset = dataset.take(batch_size * (total_examples // batch_size))

    dataset = dataset.map(lambda img, pts: parse_record_fn(img, pts, is_training),
                          num_parallel_calls=num_parallel_calls)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)

    return dataset


def get_synth_input_fn(height, width, num_channels, num_lmark):
    """Returns an input function that returns a dataset with zeroes.

    This is useful in debugging input pipeline performance, as it removes all
    elements of file reading and image preprocessing.

    Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
        tensor

    Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
    """

    def input_fn(is_training, data_dir, batch_size, *args):
        images = tf.zeros((batch_size, height, width, num_channels), tf.float32)
        labels = tf.zeros((batch_size, num_lmark, 2), tf.float32)
        return tf.data.Dataset.from_tensors((images, labels)).repeat()

    return input_fn


def dan_model_fn(features,
                 groundtruth,
                 mode,
                 stage,
                 num_lmark,
                 model_class,
                 mean_shape,
                 imgs_mean,
                 imgs_std,
                 data_format, multi_gpu=False):
    if isinstance(features, dict):
        features = features['image']

    model = model_class(num_lmark, data_format)
    resultdict = model(features,
                       stage == 1 and mode == tf.estimator.ModeKeys.TRAIN,
                       stage == 2 and mode == tf.estimator.ModeKeys.TRAIN,
                       mean_shape, imgs_mean, imgs_std)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=resultdict
        )

    loss_s1 = tf.reduce_mean(
        tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(groundtruth, resultdict['s1_ret']), -1)),
                       -1) / tf.sqrt(
            tf.reduce_sum(tf.squared_difference(tf.reduce_max(groundtruth, 1), tf.reduce_min(groundtruth, 1)), -1)))
    loss_s2 = tf.reduce_mean(
        tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(groundtruth, resultdict['s2_ret']), -1)),
                       -1) / tf.sqrt(
            tf.reduce_sum(tf.squared_difference(tf.reduce_max(groundtruth, 1), tf.reduce_min(groundtruth, 1)), -1)))

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 's1')):

        optimizer_s1 = npu_tf_optimizer(tf.train.AdamOptimizer(0.001))
        if multi_gpu:
            optimizer_s1 = npu_tf_optimizer(tf.contrib.estimator.TowerOptimizer(optimizer_s1))
        train_op_s1 = optimizer_s1.minimize(loss_s1, global_step=tf.train.get_or_create_global_step(),
                                            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 's1'))

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, 's2')):

        optimizer_s2 = npu_tf_optimizer(tf.train.AdamOptimizer(0.001))
        if multi_gpu:
            optimizer_s2 = npu_tf_optimizer(tf.contrib.estimator.TowerOptimizer(optimizer_s2))
        train_op_s2 = optimizer_s2.minimize(loss_s2, global_step=tf.train.get_or_create_global_step(),
                                            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 's2'))

    loss = loss_s1 if stage == 1 else loss_s2
    train_op = train_op_s1 if stage == 1 else train_op_s2

    if (mode == tf.estimator.ModeKeys.TRAIN or
            mode == tf.estimator.ModeKeys.EVAL):
        loss = loss_s1 if stage == 1 else loss_s2
    else:
        loss = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train_op_s1 if stage == 1 else train_op_s2
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=resultdict,
        loss=loss,
        train_op=train_op
    )


def dan_main(flags, model_function, input_function):
    if flags.multi_gpu:
        validate_batch_size_for_multi_gpu(flags.batch_size)
        model_function = tf.contrib.estimator.replicate_model_fn(model_function,
                                                                 loss_reduction=tf.losses.Reduction.MEAN)

    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=flags.inter_op_parallelism_threads,
        intra_op_parallelism_threads=flags.intra_op_parallelism_threads,
        allow_soft_placement=True)
    session_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    session_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    run_config = NPURunConfig(session_config=session_config, mix_compile_mode=True, enable_data_pre_proc=False,
                              save_checkpoints_secs=1e9)

    estimator = NPUEstimator(
        model_fn=model_function, model_dir=flags.model_dir, config=run_config,
        params={
            'dan_stage': flags.dan_stage,
            'num_lmark': flags.num_lmark,
            'data_format': flags.data_format,
            'batch_size': flags.batch_size,
            'multi_gpu': flags.multi_gpu,
        })

    def input_fn_predict():

        return input_function(False, flags.data_dir if flags.data_dir_test is not None else flags.data_dir_test,
                              flags.batch_size,
                              1, flags.num_parallel_calls, flags.multi_gpu)

    def input_fn_eval():
        return input_function(False, flags.data_dir if flags.data_dir_test is not None else flags.data_dir_test,
                              flags.batch_size,
                              1, flags.num_parallel_calls, flags.multi_gpu)

    def input_fn_train():
        return input_function(True, flags.data_dir, flags.batch_size,
                              flags.epochs_per_eval, flags.num_parallel_calls,
                              flags.multi_gpu)

    if flags.mode == tf.estimator.ModeKeys.PREDICT:
        # predict_results = estimator.predict(input_function)
        predict_results = estimator.predict(input_fn=input_fn_predict)
        save_path = os.path.join(flags.model_dir, "PREDICT")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        n = 0

        groundtruths = get_pts(flags.data_dir if flags.data_dir_test is None else flags.data_dir_test)
        errors = []

        for id, x in enumerate(predict_results):
            groundtruth = groundtruths[id]  # 真实定位点

            landmark = x['s2_ret']
            img = x['img']

            error = tests.LandmarkError(groundtruth, landmark)
            errors.append(error)

            img = np.reshape(img, (112, 112))

            landmark = landmark.astype(np.uint8)
            img = img.astype(np.uint8)

            for point in landmark:
                cv2.circle(img, point, 0, (0, 255, 0), -1)

            cv2.imwrite(os.path.join(save_path, str(n) + '.png'), img)
            n += 1
            # cv2.imshow('t', img)
            # cv2.waitKey(3000)

        avgError = np.mean(errors)
        print("Average error: {0}".format(avgError))
        tests.AUCError(errors)
        print('Save path:{}'.format(save_path))
        return

    if flags.mode == tf.estimator.ModeKeys.EVAL:
        eval_results = estimator.evaluate(input_fn=input_fn_eval, steps=flags.max_train_steps)
        print(eval_results)

    if flags.mode == tf.estimator.ModeKeys.TRAIN:
        for _ in range(flags.train_epochs // flags.epochs_per_eval):
            train_hooks = hooks_helper.get_train_hooks(["LoggingTensorHook"], batch_size=flags.batch_size)

            print('Starting a training cycle.')
            start_time = time.time()
            estimator.train(input_fn=input_fn_train,
                            max_steps=flags.max_train_steps, hooks=npu_hooks_append())
            print('epoch_time = ', (time.time() - start_time))

            print('Starting to evaluate.')
            eval_results = estimator.evaluate(input_fn=input_fn_eval,
                                              steps=flags.max_train_steps)
            print(eval_results)


class DANArgParser(argparse.ArgumentParser):
    """Arguments for configuring and running a Resnet Model.
    """

    def __init__(self):
        super(DANArgParser, self).__init__(parents=[
            parsers.BaseParser(),
            parsers.PerformanceParser(),
            parsers.ImageModelParser(),
        ])

        self.add_argument(
            "--data_dir_test", "-ddt", default=None,
            help="[default: %(default)s] The location of the test data.",
            metavar="<DD>",
        )

        self.add_argument(
            '--dan_stage', '-ds', type=int, default=1,
            choices=[1, 2],
            help='[default: %(default)s] The stage of the DAN model.'
        )

        self.add_argument(
            '--mode', '-mode', type=str, default='train',
            choices=['train', 'eval', 'predict']
        )

        self.add_argument(
            '--num_lmark', '-nlm', type=int, default=68
        )
