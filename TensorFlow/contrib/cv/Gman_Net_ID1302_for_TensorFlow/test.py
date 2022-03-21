"""
test
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image as im
from npu_bridge.npu_init import RewriterConfig, npu_config_proto
import moxing as mx
import gman_constant as constant
import gman_flags as df
from gman_log import def_logger
import gman_model as model

if not os.path.exists(df.FLAGS.train_dir):
    os.makedirs(df.FLAGS.train_dir)
    os.makedirs(df.FLAGS.clear_result_images_dir)
    print('make dir: {}\t and {}'.format(df.FLAGS.train_dir + '/', df.FLAGS.clear_result_images_dir))
mx.file.copy_parallel(df.FLAGS.checkpoint_dir_obs, df.FLAGS.checkpoint_dir)
print('copy saved model to obs: {}.'.format(df.FLAGS.train_url))
logger = def_logger(df.FLAGS.clear_result_images_dir + "log.txt")


def cal_psnr(im1, im2):
    """

    Args:
        im1:
        im2:

    Returns:

    """
    mse = ((im1.astype(np.float32) - im2.astype(np.float32)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def evaluate_one_by_one():
    """

    Returns:

    """
    # A list used to save all psnr and ssim.
    psnr_list = []
    # Read all hazed images indexes and clear images from directory

    graph = tf.Graph()
    with graph.as_default():
        sess_config = tf.ConfigProto()
        custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # # set precision mode
        # custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes(
        #     'allow_mix_precision')
        # dump path
        # custom_op.parameter_map['dump_path'].s = tf.compat.as_bytes('/cache/saveModels/')
        # # set dump debug
        # custom_op.parameter_map['enable_dump_debug'].b = True
        # custom_op.parameter_map['dump_debug_mode'].s = tf.compat.as_bytes('all')
        # custom_op.parameter_map["profiling_mode"].b = True
        # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
        #     '{"output":"/cache/saveModels","task_trace":"on"}')
        sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        # ########################################################################
        # ########################Load images from disk##############################
        # ########################################################################
        hazed_image_placeholder = tf.placeholder(tf.float32,
                                                 shape=[1, df.FLAGS.input_image_height, df.FLAGS.input_image_width,
                                                        constant.RGB_CHANNEL])

        # ########################################################################
        # ###################Restore model and do evaluations#####################
        # ########################################################################
        gman = model.Gman()
        logist = gman.inference(hazed_image_placeholder, batch_size=1, h=df.FLAGS.input_image_height,
                                w=df.FLAGS.input_image_width)
        variable_averages = tf.train.ExponentialMovingAverage(
            constant.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # saver, train_op, hazed_image, clear_image_arr, hazed_images_obj, placeholder, psnr_list, ssim_list, h, w
        with tf.Session(graph=graph, config=npu_config_proto(config_proto=sess_config)) as sess:
            ckpt = tf.train.get_checkpoint_state(df.FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                return
            for i in range(14000):
                # Read image from files and append them to the list
                hazed_image = im.open(df.FLAGS.haze_test_images_dir + '{}.jpg'.format(i))
                hazed_image = hazed_image.convert('RGB')
                clear_image = im.open(df.FLAGS.clear_test_images_dir + '{}.jpg'.format(i))
                clear_image = clear_image.convert('RGB')
                clear_image_arr = np.array(clear_image)
                hazed_image_arr = np.array(hazed_image)
                float_hazed_image = hazed_image_arr.astype('float32') / 255.0
                float_clear_image = clear_image_arr.astype('float32') / 255.0

                start = time.time()
                prediction = sess.run([logist], feed_dict={hazed_image_placeholder: [float_hazed_image]})
                duration = time.time() - start
                # Run the session and get the prediction of one clear image
                dehazed_image = write_images_to_file(prediction, i, df.FLAGS.input_image_height,
                                                     df.FLAGS.input_image_width)
                psnr_value = cal_psnr(dehazed_image, clear_image_arr)
                psnr_list.append(psnr_value)
                logger.info(
                    '-------------------------------------------------------------------------------------------------------------------------------')
                format_str = 'image: %d PSNR: %f; (%.4f seconds)'
                logger.info(format_str % (i, psnr_value, duration))
                logger.info(
                    '-------------------------------------------------------------------------------------------------------------------------------')

    if not df.FLAGS.eval_only_haze:
        psnr_avg = np.mean(psnr_list)
        format_str = 'Average PSNR: %5f'
        logger.info(format_str % psnr_avg)


# def normalization(data):
#     _range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / _range

def write_images_to_file(logist, i, height, width):
    """

    Args:
        logist:
        i:
        height:
        width:

    Returns:

    """
    array = np.reshape(logist[0], newshape=[height, width, constant.RGB_CHANNEL])
    # array = normalization(array)*255.0
    array = array * 255.0
    result_image = im.fromarray(array.astype('uint8'))
    result_image.save(df.FLAGS.clear_result_images_dir + '{}.jpg'.format(i))
    return array


if __name__ == '__main__':
    evaluate_one_by_one()
    mx.file.copy_parallel('/cache/saveModels', df.FLAGS.train_url)
    print('copy saved results to obs: {}.'.format(df.FLAGS.train_url))
