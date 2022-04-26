#!/usr/bin/env python
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
# Copyright 2022 Huawei Technologies Co., Ltd
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

from npu_bridge.npu_init import *



import tensorflow as tf
import tensorflow.contrib.slim as slim

import utils
import os
import numpy as np
import argparse
import network
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"]="0"

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default = 256, type = int)
    parser.add_argument("--batch_size", default = 16, type = int)
    parser.add_argument("--total_iter", default = 50000, type = int)
    parser.add_argument("--adv_train_lr", default = 2e-4, type = float)
    parser.add_argument("--gpu_fraction", default = 0.5, type = float)
    parser.add_argument("--data_path", default='/cache/dataset', type=str)

    args = parser.parse_args()

    return args



def train(args):
    SAVE_DIR = "./pretrain"
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    input_photo = tf.placeholder(tf.float32, [args.batch_size,
                                args.patch_size, args.patch_size, 3])

    output = network.unet_generator(input_photo)

    recon_loss = tf.reduce_mean(tf.losses.absolute_difference(input_photo, output))

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'gene' in var.name]
    loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
                                                             decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):

        optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)
        optim = NPULossScaleOptimizer(optim, loss_scale_manager).minimize(recon_loss, var_list=gene_vars)

    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    '''
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    config = tf.ConfigProto(gpu_options=gpu_options)
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp32")

    sess = tf.Session(config=config)
    saver = tf.train.Saver(var_list=gene_vars, max_to_keep=20)

    with tf.device('/cpu:0'):

        sess.run(tf.global_variables_initializer())
        face_photo_dir = os.path.join(args.data_path, 'face_photo')
        face_photo_list = utils.load_image_list(face_photo_dir)
        scenery_photo_dir = os.path.join(args.data_path, 'scenery_photo')
        scenery_photo_list = utils.load_image_list(scenery_photo_dir)


        for total_iter in tqdm(range(args.total_iter)):

            if np.mod(total_iter, 5) == 0:
                photo_batch = utils.next_batch(face_photo_list, args.batch_size)
            else:
                photo_batch = utils.next_batch(scenery_photo_list, args.batch_size)
            lossScale = tf.get_default_graph().get_tensor_by_name("loss_scale:0")
            _, _, r_loss = sess.run([lossScale, optim, recon_loss], feed_dict={input_photo: photo_batch})

            if np.mod(total_iter+1, 50) == 0:

                print('pretrain, iter: {}, recon_loss: {}'.format(total_iter, r_loss))
                if np.mod(total_iter+1, 500 ) == 0:
                    saver.save(sess, SAVE_DIR+'/saved_models/model',
                               write_meta_graph=False, global_step=total_iter)

                    photo_face = utils.next_batch(face_photo_list, args.batch_size)
                    photo_scenery = utils.next_batch(scenery_photo_list, args.batch_size)

                    result_face = sess.run(output, feed_dict={input_photo: photo_face})

                    result_scenery = sess.run(output, feed_dict={input_photo: photo_scenery})

                    utils.write_batch_image(result_face, SAVE_DIR+'/images',
                                            str(total_iter)+'_face_result.jpg', 4)
                    utils.write_batch_image(photo_face,  SAVE_DIR+'/images',
                                            str(total_iter)+'_face_photo.jpg', 4)
                    utils.write_batch_image(result_scenery,  SAVE_DIR+'/images',
                                            str(total_iter)+'_scenery_result.jpg', 4)
                    utils.write_batch_image(photo_scenery,  SAVE_DIR+'/images',
                                            str(total_iter)+'_scenery_photo.jpg', 4)


if __name__ == '__main__':

    args = arg_parser()
    train(args)

