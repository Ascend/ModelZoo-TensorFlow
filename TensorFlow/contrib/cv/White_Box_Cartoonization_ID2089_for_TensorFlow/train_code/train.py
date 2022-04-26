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

import tensorflow as tf
from npu_bridge.npu_init import *

import tensorflow.contrib.slim as slim

import utils
import os
import moxing as mox
import numpy as np
import argparse
import network
import loss
import time

# from tqdm import tqdm
from guided_filter import guided_filter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# modelarts modification------------------------------
CACHE_TRAINING_URL = "/cache/training/"
SAVE_DIR = CACHE_TRAINING_URL + "train_cartoon"
REAL_PATH = '/cache/dataset'

if not os.path.isdir(CACHE_TRAINING_URL):
    os.makedirs(CACHE_TRAINING_URL)


# modelarts modification------------------------------


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default=256, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--total_iter", default=100000, type=int)
    parser.add_argument("--adv_train_lr", default=2e-4, type=float)
    parser.add_argument("--gpu_fraction", default=0.5, type=float)
    parser.add_argument("--save_dir", default='train_cartoon', type=str)
    parser.add_argument("--use_enhance", default=False)
    parser.add_argument("--data_path", default='/cache/dataset', type=str)
    parser.add_argument("--output_path", default='/cache/output', type=str)

    args = parser.parse_args()

    return args


def train(args):
    # modelarts modification------------------------------
    if not os.path.exists(REAL_PATH):
        os.makedirs(REAL_PATH, 0o755)
    mox.file.copy_parallel(args.data_path, REAL_PATH)
    print("training data finish copy to %s." % REAL_PATH)

    input_photo = tf.placeholder(tf.float32, [args.batch_size,
                                              args.patch_size, args.patch_size, 3])
    input_superpixel = tf.placeholder(tf.float32, [args.batch_size,
                                                   args.patch_size, args.patch_size, 3])
    input_cartoon = tf.placeholder(tf.float32, [args.batch_size,
                                                args.patch_size, args.patch_size, 3])

    output = network.unet_generator(input_photo)
    output = guided_filter(input_photo, output, r=1)

    blur_fake = guided_filter(output, output, r=5, eps=2e-1)
    blur_cartoon = guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)

    gray_fake, gray_cartoon = utils.color_shift(output, input_cartoon)

    d_loss_gray, g_loss_gray = loss.lsgan_loss(network.disc_sn, gray_cartoon, gray_fake,
                                               scale=1, patch=True, name='disc_gray')
    d_loss_blur, g_loss_blur = loss.lsgan_loss(network.disc_sn, blur_cartoon, blur_fake,
                                               scale=1, patch=True, name='disc_blur')

    vgg_path = os.path.join(REAL_PATH, 'vgg19_no_fc.npy')
    vgg_model = loss.Vgg19(vgg_path)
    vgg_photo = vgg_model.build_conv4_4(input_photo)
    vgg_output = vgg_model.build_conv4_4(output)
    vgg_superpixel = vgg_model.build_conv4_4(input_superpixel)
    h, w, c = vgg_photo.get_shape().as_list()[1:]

    photo_loss = tf.reduce_mean(tf.losses.absolute_difference(vgg_photo, vgg_output)) / (h * w * c)
    superpixel_loss = tf.reduce_mean(tf.losses.absolute_difference \
                                         (vgg_superpixel, vgg_output)) / (h * w * c)
    recon_loss = photo_loss + superpixel_loss
    tv_loss = loss.total_variation_loss(output)

    g_loss_total = 1e4 * tv_loss + 1e-1 * g_loss_blur + g_loss_gray + 2e2 * recon_loss
    d_loss_total = d_loss_blur + d_loss_gray

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'gene' in var.name]
    disc_vars = [var for var in all_vars if 'disc' in var.name]

    tf.summary.scalar('tv_loss', tv_loss)
    tf.summary.scalar('photo_loss', photo_loss)
    tf.summary.scalar('superpixel_loss', superpixel_loss)
    tf.summary.scalar('recon_loss', recon_loss)
    tf.summary.scalar('d_loss_gray', d_loss_gray)
    tf.summary.scalar('g_loss_gray', g_loss_gray)
    tf.summary.scalar('d_loss_blur', d_loss_blur)
    tf.summary.scalar('g_loss_blur', g_loss_blur)
    tf.summary.scalar('d_loss_total', d_loss_total)
    tf.summary.scalar('g_loss_total', g_loss_total)

    loss_scale_manager_g = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
                                                             decr_every_n_nan_or_inf=2, decr_ratio=0.5)
    loss_scale_manager_d = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
                                                             decr_every_n_nan_or_inf=2, decr_ratio=0.5)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        g_optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)
        g_optim = NPULossScaleOptimizer(g_optim, loss_scale_manager_g).minimize(g_loss_total, var_list=gene_vars)

        d_optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99)
        d_optim = NPULossScaleOptimizer(d_optim, loss_scale_manager_d).minimize(d_loss_total, var_list=disc_vars)
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

    train_writer = tf.summary.FileWriter(SAVE_DIR + '/train_log')
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(var_list=gene_vars, max_to_keep=20)

    with tf.device('/cpu:0'):

        sess.run([tf.global_variables_initializer()])
        saver.restore(sess, tf.train.latest_checkpoint('pretrain/saved_models'))

        face_photo_dir = os.path.join(REAL_PATH, 'face_photo')
        print("face photo dir = ", face_photo_dir)
        face_photo_list = utils.load_image_list(face_photo_dir)
        scenery_photo_dir = os.path.join(REAL_PATH, 'scenery_photo')
        scenery_photo_list = utils.load_image_list(scenery_photo_dir)

        face_cartoon_dir = os.path.join(REAL_PATH, 'face_cartoon')
        face_cartoon_list = utils.load_image_list(face_cartoon_dir)
        scenery_cartoon_dir = os.path.join(REAL_PATH, 'scenery_cartoon')
        scenery_cartoon_list = utils.load_image_list(scenery_cartoon_dir)

        for total_iter in range(args.total_iter):
            if np.mod(total_iter, 5) == 0:
                photo_batch = utils.next_batch(face_photo_list, args.batch_size)
                cartoon_batch = utils.next_batch(face_cartoon_list, args.batch_size)
            else:
                photo_batch = utils.next_batch(scenery_photo_list, args.batch_size)
                cartoon_batch = utils.next_batch(scenery_cartoon_list, args.batch_size)

            start_time = time.time()
            inter_out = sess.run(output, feed_dict={input_photo: photo_batch,
                                                    input_superpixel: photo_batch,
                                                    input_cartoon: cartoon_batch})

            '''
            adaptive coloring has to be applied with the clip_by_value 
            in the last layer of generator network, which is not very stable.
            to stabiliy reproduce our results, please use power=1.0
            and comment the clip_by_value function in the network.py first
            If this works, then try to use adaptive color with clip_by_value.
            '''
            if args.use_enhance:
                superpixel_batch = utils.selective_adacolor(inter_out, power=1.2)
            else:
                superpixel_batch = utils.simple_superpixel(inter_out, seg_num=200)

            lossScale = tf.get_default_graph().get_tensor_by_name("loss_scale:0")
            l_s_g, _, g_loss, r_loss = sess.run([lossScale, g_optim, g_loss_total, recon_loss],
                                                feed_dict={input_photo: photo_batch, input_superpixel: superpixel_batch,
                                                           input_cartoon: cartoon_batch})

            l_s_d, _, d_loss, train_info = sess.run([lossScale, d_optim, d_loss_total, summary_op],
                                                    feed_dict={input_photo: photo_batch,
                                                               input_superpixel: superpixel_batch,
                                                               input_cartoon: cartoon_batch})

            duration = (time.time() - start_time)
            ms_per_batch = float(duration)
            print("Iter: %d/%d , time_per_step %.3f" % (total_iter, args.total_iter, ms_per_batch))

            train_writer.add_summary(train_info, total_iter)
            if np.mod(total_iter + 1, 50) == 0:

                # print('Iter: {}, loss_scale g: {}, loss_scale d: {}'.format(total_iter, l_s_g, l_s_d))
                print('Iter: {}, d_loss: {}, g_loss: {}, recon_loss: {}'. \
                      format(total_iter, d_loss, g_loss, r_loss))
                if np.mod(total_iter + 1, 500) == 0:
                    saver.save(sess, SAVE_DIR + '/saved_models/model',
                               write_meta_graph=False, global_step=total_iter)

                    photo_face = utils.next_batch(face_photo_list, args.batch_size)
                    cartoon_face = utils.next_batch(face_cartoon_list, args.batch_size)
                    photo_scenery = utils.next_batch(scenery_photo_list, args.batch_size)
                    cartoon_scenery = utils.next_batch(scenery_cartoon_list, args.batch_size)

                    result_face = sess.run(output, feed_dict={input_photo: photo_face,
                                                              input_superpixel: photo_face,
                                                              input_cartoon: cartoon_face})

                    result_scenery = sess.run(output, feed_dict={input_photo: photo_scenery,
                                                                 input_superpixel: photo_scenery,
                                                                 input_cartoon: cartoon_scenery})

                    utils.write_batch_image(result_face, SAVE_DIR + '/images',
                                            str(total_iter) + '_face_result.jpg', 4)
                    utils.write_batch_image(photo_face, SAVE_DIR + '/images',
                                            str(total_iter) + '_face_photo.jpg', 4)

                    utils.write_batch_image(result_scenery, SAVE_DIR + '/images',
                                            str(total_iter) + '_scenery_result.jpg', 4)
                    utils.write_batch_image(photo_scenery, SAVE_DIR + '/images',
                                            str(total_iter) + '_scenery_photo.jpg', 4)

    mox.file.copy_parallel(CACHE_TRAINING_URL, args.output_path)
    sess.close()


if __name__ == '__main__':
    args = arg_parser()
    train(args)  
