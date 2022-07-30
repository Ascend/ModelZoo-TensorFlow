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
# ============================================================================

from __future__ import division
from npu_bridge.npu_init import *
import os
import random
import numpy as np
import tensorflow as tf


class DataLoader(object):
    def __init__(self, trainable=True, **config):
        self.config = config
        self.dataset_dir = self.config['dataset']['root_dir']
        self.batch_size = np.int(self.config['model']['batch_size'])
        self.img_height = np.int(self.config['dataset']['image_height'])
        self.img_width = np.int(self.config['dataset']['image_width'])
        self.num_source = np.int(self.config['model']['num_source']) - 1
        self.num_scales = np.int(self.config['model']['num_scales'])

        self.trainable = trainable

    def load_batch(self):
        """Load a batch of training instances.
        """
        file_list = self.format_file_list(self.dataset_dir, 'train' if self.trainable else 'val')

        self.steps_per_epoch = int(
            len(file_list['image_file_list']) // self.batch_size)

        def parse_image(image_path):
            image_contents = tf.io.read_file(image_path)
            image_seq = tf.image.decode_jpeg(image_contents)
            # [H, W, 3] and [H, W, 3 * num_source]
            tgt_image, src_image_stack = \
                self.unpack_image_sequence(
                    image_seq, self.img_height, self.img_width, self.num_source)

            return src_image_stack, tgt_image

        def parse_cam(line):
            rec_def = []
            for i in range(9):
                rec_def.append([1.])
            raw_cam_vec = tf.decode_csv(line,
                                        record_defaults=rec_def)
            raw_cam_vec = tf.stack(raw_cam_vec)
            intrinsics = tf.reshape(raw_cam_vec, [3, 3])
            return intrinsics

        ds_image = tf.data.Dataset.from_tensor_slices(file_list['image_file_list'])
        ds_image = ds_image.repeat()
        ds_image = ds_image.map(lambda image_path: parse_image(image_path), num_parallel_calls=64)
        ds_image = ds_image.batch(self.batch_size, drop_remainder=True).prefetch(64)
        ds_image_iterator = ds_image.make_initializable_iterator()
        src_image_stack, tgt_image = ds_image_iterator.get_next()  # (16,192,640,6) & (16,192,640,3)

        ds_cam = tf.data.TextLineDataset(file_list['cam_file_list'])
        ds_cam = ds_cam.repeat()
        ds_cam = ds_cam.map(parse_cam, num_parallel_calls=64)
        ds_cam = ds_cam.batch(self.batch_size, drop_remainder=True).prefetch(64)
        ds_cam_iterator = ds_cam.make_initializable_iterator()
        intrinsics = ds_cam_iterator.get_next()   # (16,3,3)

        # Data augmentation
        image_all = tf.concat([tgt_image, src_image_stack], axis=3)
        image_all, image_all_aug = self.data_augmentation(
            image_all)

        tgt_image = image_all[:, :, :, :3]
        src_image_stack = image_all[:, :, :, 3:]

        tgt_image_aug = image_all_aug[:, :, :, :3]
        src_image_stack_aug = image_all_aug[:, :, :, 3:]
        intrinsics = self.get_multi_scale_intrinsics(
            intrinsics, self.num_scales)
        return tgt_image, src_image_stack, tgt_image_aug, src_image_stack_aug, intrinsics, ds_image_iterator, ds_cam_iterator

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0., 0., 1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics


    def data_augmentation(self, im):
        def random_flip(im):
            def flip_one(sim):
                do_flip = tf.random_uniform([], 0, 1)
                return tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(sim), lambda: sim)

            im = tf.map_fn(lambda sim: flip_one(sim), im)
            return im

        def augment_image_properties(im):
            # random brightness
            brightness_seed = random.randint(0, 2 ** 31 - 1)
            im = tf.image.random_brightness(im, 0.2, brightness_seed)

            contrast_seed = random.randint(0, 2 ** 31 - 1)
            im = tf.image.random_contrast(im, 0.8, 1.2, contrast_seed)

            num_img = np.int(im.get_shape().as_list()[-1] // 3)

            saturation_im_list = []
            saturation_factor = random.uniform(0.8,
                                               1.2)  
            for i in range(num_img):
                saturation_im_list.append(tf.image.adjust_saturation(im[:, :, 3 * i: 3 * (i + 1)], saturation_factor))
            im = tf.concat(saturation_im_list, axis=2)

            hue_im_list = []
            hue_delta = random.uniform(-0.1, 0.1)  
            for i in range(num_img):
                hue_im_list.append(tf.image.adjust_hue(im[:, :, 3 * i: 3 * (i + 1)], hue_delta))
            im = tf.concat(hue_im_list, axis=2)
            return im

        def random_augmentation(im):
            def augmentation_one(sim):
                do_aug = tf.random_uniform([], 0, 1)
                return tf.cond(do_aug > 0.5, lambda: augment_image_properties(sim), lambda: sim)

            im = tf.map_fn(lambda sim: augmentation_one(sim), im)
            return im

        im = random_flip(im)
        im_aug = random_augmentation(im)
        return im, im_aug

    def format_file_list(self, data_root, split):
        with open(data_root + '/%s.txt' % split, 'r') as f:
            frames = f.readlines()
        subfolders = [x.split(' ')[0] for x in frames]
        frame_ids = [x.split(' ')[1][:-1] for x in frames]
        image_file_list = [os.path.join(data_root, subfolders[i],
                                        frame_ids[i] + '.jpg') for i in range(len(frames))]
        cam_file_list = [os.path.join(data_root, subfolders[i],
                                      frame_ids[i] + '_cam.txt') for i in range(len(frames))]
        all_list = {}
        all_list['image_file_list'] = image_file_list
        all_list['cam_file_list'] = cam_file_list
        return all_list

    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        tgt_start_idx = int(img_width * (num_source // 2))
        # [h, w, 3]
        tgt_image = tf.slice(image_seq,
                             [0, tgt_start_idx, 0],
                             [-1, img_width, -1])
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0],
                               [-1, int(img_width * (num_source // 2)), -1])
        src_image_2 = tf.slice(image_seq,
                               [0, int(tgt_start_idx + img_width), 0],
                               [-1, int(img_width * (num_source // 2)), -1])

        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                              [0, i * img_width, 0],
                                              [-1, img_width, -1])
                                     for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height,
                                   img_width,
                                   num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])
        return tgt_image, src_image_stack

    def batch_unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        tgt_start_idx = int(img_width * (num_source // 2))
        tgt_image = tf.slice(image_seq,
                             [0, 0, tgt_start_idx, 0],
                             [-1, -1, img_width, -1])
        src_image_1 = tf.slice(image_seq,
                               [0, 0, 0, 0],
                               [-1, -1, int(img_width * (num_source // 2)), -1])
        src_image_2 = tf.slice(image_seq,
                               [0, 0, int(tgt_start_idx + img_width), 0],
                               [-1, -1, int(img_width * (num_source // 2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=2)
        src_image_stack = tf.concat([tf.slice(src_image_seq,
                                              [0, 0, i * img_width, 0],
                                              [-1, -1, img_width, -1])
                                     for i in range(num_source)], axis=3)
        return tgt_image, src_image_stack

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        for s in range(num_scales):
            fx = intrinsics[:, 0, 0] / (2 ** s)
            fy = intrinsics[:, 1, 1] / (2 ** s)
            cx = intrinsics[:, 0, 2] / (2 ** s)
            cy = intrinsics[:, 1, 2] / (2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale

