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
"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import time

import cv2
import numpy as np
import tensorflow as tf

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

gt_path = './eval_data/gt.png'


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', default='./eval_data/test_img.jpg',
                        type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', default='./pretrained_model/ckpt/tusimple_lanenet.ckpt',
                        type=str, help='The model weights path')
    parser.add_argument('--with_lane_fit', type=args_str2bool, help='If need to do lane fit', default=False)

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(image_path, gt_path, weights_path, with_lane_fit=False):
    """

    :param image_path:
    :param weights_path:
    :param with_lane_fit:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    LOG.info('Start reading image and preprocessing')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0
    LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    input_tensor_0 = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    input_tensor = tf.identity(input_tensor_0, name='input_tensor')
    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    with tf.variable_scope('lanenet/'):
        binary_seg_ret = tf.identity(binary_seg_ret, name='binary_seg_out')
        instance_seg_ret = tf.identity(instance_seg_ret, name='instance_seg_out')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    sess = tf.Session()

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        image = np.expand_dims(image, axis=0)
        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: image}
        )

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
            with_lane_fit=True,
            data_source='tusimple'
        )
        mask_image = postprocess_result['mask_image']
        src_image = postprocess_result['source_image']

        # -------------- 计算准确率 ------------------ #
        gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        gt_trans = cv2.resize(gt, (512, 256), interpolation=cv2.INTER_LINEAR)

        gt_gray = cv2.cvtColor(gt_trans, cv2.COLOR_BGR2GRAY)
        mask_image_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        WIDTH = mask_image_gray.shape[0]
        HIGTH = mask_image_gray.shape[1]
        tp_count = 0
        tn_count = 0
        for i in range(WIDTH):
            for j in range(HIGTH):
                if mask_image_gray[i, j] != 0 and gt_gray[i, j] != 0:
                    tp_count = tp_count + 1
                if mask_image_gray[i, j] == 0 and gt_gray[i, j] == 0:
                    tn_count = tn_count + 1
        Accuracy = (int(tp_count) + int(tn_count)) / (int(WIDTH) * int(HIGTH))

        print("\n# Metric_ckpt "
              "\n     Accuracy：{:.3f}".format(Accuracy))

        cv2.imwrite('./eval_output/mask_ckpt.jpg', mask_image)
        cv2.imwrite('./eval_output/src_ckpt.jpg', src_image)

        saver.save(sess, './eval_ckpt/eval.ckpt')

    sess.close()

    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    args = init_args()

    test_lanenet(args.image_path, gt_path, args.weights_path, with_lane_fit=args.with_lane_fit)
