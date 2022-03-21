"""
eval
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
from npu_bridge.npu_init import RewriterConfig, npu_config_proto
import moxing as mx
import gman_flags as df
import os

import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image as im
# from skimage.measure import compare_ssim

import gman_constant as constant
from datasets import parse_record, Image, image_input_eval, find_corres_clear_image_rgb
import gman_log
import gman_model as model

if not os.path.exists(df.FLAGS.train_dir):
    os.makedirs(df.FLAGS.train_dir)
    os.makedirs(df.FLAGS.clear_result_images_dir)
    print('make dir: {}\t and {}'.format(df.FLAGS.train_dir + '/', df.FLAGS.clear_result_images_dir))
mx.file.copy_parallel(df.FLAGS.checkpoint_dir_obs, df.FLAGS.checkpoint_dir)
print('copy saved model to obs: {}.'.format(df.FLAGS.train_url))
logger = gman_log.def_logger(df.FLAGS.clear_result_images_dir + "log.txt")
# Frames used to save clear training image information
_clear_test_file_names = []
_clear_test_img_list = []
_clear_test_directory = {}
# Frames used to save hazed training image information
_hazed_test_file_names = []
_hazed_test_img_list = []

_hazed_test_A_dict = {}
_hazed_test_beta_dict = {}


def tf_psnr(im1, im2):
    """

    Args:
        im1:
        im2:

    Returns:

    """
    # assert pixel value range is 0-1
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))


def cal_psnr(im1, im2):
    """

    Args:
        im1:
        im2:

    Returns:

    """
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def eval_once(graph, saver, train_op, hazed_image, clear_image, hazed_images_obj, placeholder, psnr_list, h, w):
    """

    Args:
        graph:
        saver:
        train_op:
        hazed_image:
        clear_image:
        hazed_images_obj:
        placeholder:
        psnr_list:
        h:
        w:

    Returns:

    """
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
    with tf.Session(graph=graph, config=npu_config_proto(config_proto=sess_config)) as sess:
        ckpt = tf.train.get_checkpoint_state(df.FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return
        start = time.time()
        prediction = sess.run([train_op], feed_dict={placeholder: [hazed_image]})
        duration = time.time() - start
        # Run the session and get the prediction of one clear image
        dehazed_image = write_images_to_file(prediction, hazed_images_obj, h, w, sess)
        if not df.FLAGS.eval_only_haze:
            psnr_value = cal_psnr(dehazed_image, clear_image)
            # ssim_value = compare_ssim(np.uint8(dehazed_image), np.uint8(clear_image), multichannel=True)
            # ssim_list.append(ssim_value)
            psnr_list.append(psnr_value)
            logger.info(
                '--------------------------------------------------------------')
            format_str = 'image: %s PSNR: %f; (%.4f seconds)'
            logger.info(format_str % (hazed_images_obj.path, psnr_value, duration))
            logger.info(
                '--------------------------------------------------------------')
        else:
            print(
                '---------------------------------------------------------------')
            format_str = 'image: %s (%.4f seconds)'
            logger.info(format_str % (hazed_images_obj.path, duration))
            print(
                '----------------------------------------------------------------')
    sess.close()


def evaluate_one_by_one():
    """

    Returns:

    """
    # A list used to save all psnr and ssim.
    psnr_list = []
    # ssim_list = []
    # Read all hazed images indexes and clear images from directory
    if not df.FLAGS.eval_only_haze:
        image_input_eval(df.FLAGS.clear_test_images_dir, _clear_test_file_names, _clear_test_img_list,
                         _clear_test_directory, clear_image=True)
        if len(_clear_test_img_list) == 0:
            raise RuntimeError("No image found! Please supply clear images for training or eval ")
    # Hazed training image pre-process
    image_input_eval(df.FLAGS.haze_test_images_dir, _hazed_test_file_names, _hazed_test_img_list,
                     clear_dict=None, clear_image=False)
    if len(_hazed_test_img_list) == 0:
        raise RuntimeError("No image found! Please supply hazed images for training or eval ")

    for image in _hazed_test_img_list[:400]:
        graph = tf.Graph()
        with graph.as_default():
            # ########################################################################
            # ########################Load images from disk##############################
            # ########################################################################
            # Read image from files and append them to the list
            hazed_image = im.open(image.path)
            hazed_image = hazed_image.convert('RGB')
            shape = np.shape(hazed_image)
            hazed_image_placeholder = tf.placeholder(tf.float32,
                                                     shape=[constant.SINGLE_IMAGE_NUMBER, shape[0], shape[1],
                                                            constant.RGB_CHANNEL])
            hazed_image_arr = np.array(hazed_image)
            float_hazed_image = hazed_image_arr.astype('float32') / 255
            if not df.FLAGS.eval_only_haze:
                clear_image = find_corres_clear_image_rgb(image, _clear_test_directory)
                clear_image_arr = np.array(clear_image)

            # ########################################################################
            # ###################Restore model and do evaluations#####################
            # ########################################################################
            gman = model.Gman()
            logist = gman.inference(hazed_image_placeholder, batch_size=1, h=shape[0], w=shape[1])
            variable_averages = tf.train.ExponentialMovingAverage(
                constant.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            # saver, train_op, hazed_image, clear_image_arr, hazed_images_obj, placeholder, psnr_list, ssim_list, h, w
            if not df.FLAGS.eval_only_haze:
                eval_once(graph, saver, logist, float_hazed_image, clear_image_arr, image, hazed_image_placeholder,
                          psnr_list, shape[0], shape[1])
            else:
                eval_once(graph, saver, logist, float_hazed_image, None, image, hazed_image_placeholder,
                          psnr_list, shape[0], shape[1])

    if not df.FLAGS.eval_only_haze:
        psnr_avg = cal_average(psnr_list)
        format_str = 'Average PSNR: %5f'
        logger.info(format_str % psnr_avg)
        # ssim_avg = cal_average(ssim_list)
        # format_str = 'Average SSIM: %5f'
        # logger.info(format_str % ssim_avg)


def cal_average(result_list):
    """

    Args:
        result_list:

    Returns:

    """
    sum_psnr = sum(result_list)
    return sum_psnr / len(result_list)


def write_images_to_file(logist, image, height, width, sess):
    """

    Args:
        logist:
        image:
        height:
        width:
        sess:

    Returns:

    """
    array = np.reshape(logist[0], newshape=[height, width, constant.RGB_CHANNEL])
    array *= 255
    result_image = tf.saturate_cast(array, tf.uint8)
    arr1 = sess.run(result_image)
    result_image = im.fromarray(arr1, 'RGB')
    image_name_base = image.image_index
    result_image.save(df.FLAGS.clear_result_images_dir + image_name_base + "_" + str(time.time()) + '_pred.png')
    return array


if __name__ == '__main__':
    evaluate_one_by_one()
    mx.file.copy_parallel('/cache/saveModels', df.FLAGS.train_url)
    print('copy saved results to obs: {}.'.format(df.FLAGS.train_url))
