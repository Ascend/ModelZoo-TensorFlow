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
# Mostly based on the code written by Tinghui Zhou & Clement Godard: 
# https://github.com/tinghuiz/SfMLearner/blob/master/data_loader.py
# https://github.com/mrharicot/monodepth/blob/master/monodepth_dataloader.py
from __future__ import division
from npu_bridge.npu_init import *
import os
import random
import tensorflow as tf

class DataLoader(object):
    def __init__(self, opt=None):
        self.opt = opt

    def load_train_batch(self):
        """Load a batch of training instances.
        """
        opt = self.opt

        # Load the list of training files into queues
        file_list = self.format_file_list(opt.dataset_dir, 'train')

        def parse_image(image_path):

            #cam_paths_queue = tf.train.string_input_producer(
                #file_list['cam_file_list'], shuffle=False)

            # Load images
            image_contents = tf.io.read_file(image_path)
            image_seq = tf.image.decode_jpeg(image_contents)

            tgt_image, src_image_stack = \
                self.unpack_image_sequence(
                    image_seq, opt.img_height, opt.img_width, opt.num_source)

            return src_image_stack, tgt_image

        def parse_cam(line):
            rec_def = []
            for i in range(9):
                rec_def.append([1.])
            raw_cam_vec = tf.decode_csv(line, record_defaults=rec_def)
            raw_cam_vec = tf.stack(raw_cam_vec)
            intrinsics = tf.reshape(raw_cam_vec, [3, 3])
            return intrinsics

        ds_image = tf.data.Dataset.from_tensor_slices(file_list['image_file_list']).repeat()
        ds_image = ds_image.map(lambda image_path: parse_image(image_path), num_parallel_calls=64)
        ds_image = ds_image.batch(opt.batch_size, drop_remainder=True).prefetch(64)
        ds_image_iterator = ds_image.make_initializable_iterator()
        src_image_stack, tgt_image = ds_image_iterator.get_next()

        ds_cam = tf.data.TextLineDataset(file_list['cam_file_list']).repeat()
        ds_cam = ds_cam.map(parse_cam, num_parallel_calls=64)
        ds_cam = ds_cam.batch(opt.batch_size, drop_remainder=True).prefetch(64)
        ds_cam_iterator = ds_cam.make_initializable_iterator()
        intrinsics = ds_cam_iterator.get_next()

        # Data augmentation
        image_all = tf.concat([tgt_image, src_image_stack], axis=3)
        image_all, intrinsics = self.data_augmentation(
            image_all, intrinsics, opt.img_height, opt.img_width)
        tgt_image = image_all[:, :, :, :3]
        src_image_stack = image_all[:, :, :, 3:]
        intrinsics = self.get_multi_scale_intrinsics(
            intrinsics, opt.num_scales)
        return tgt_image, src_image_stack, intrinsics, ds_image_iterator, ds_cam_iterator

    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics

    def data_augmentation(self, im, intrinsics, out_h, out_w):
        # Random scaling
        def random_scaling(im, intrinsics):
            batch_size, in_h, in_w, _ = im.get_shape().as_list()
            scaling = tf.random_uniform([2], 1, 1.15)
            x_scaling = scaling[0]
            y_scaling = scaling[1]
            out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
            out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
            im = tf.image.resize_area(im, [out_h, out_w])
            fx = intrinsics[:,0,0] * x_scaling
            fy = intrinsics[:,1,1] * y_scaling
            cx = intrinsics[:,0,2] * x_scaling
            cy = intrinsics[:,1,2] * y_scaling
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random cropping
        def random_cropping(im, intrinsics, out_h, out_w):
            # batch_size, in_h, in_w, _ = im.get_shape().as_list()
            batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
            offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
            offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
            im = tf.image.crop_to_bounding_box(
                im, offset_y, offset_x, out_h, out_w)
            fx = intrinsics[:,0,0]
            fy = intrinsics[:,1,1]
            cx = intrinsics[:,0,2] - tf.cast(offset_x, dtype=tf.float32)
            cy = intrinsics[:,1,2] - tf.cast(offset_y, dtype=tf.float32)
            intrinsics = self.make_intrinsics_matrix(fx, fy, cx, cy)
            return im, intrinsics

        # Random coloring
        def random_coloring(im):
            batch_size, in_h, in_w, in_c = im.get_shape().as_list()
            im_f = tf.image.convert_image_dtype(im, tf.float32)

            # randomly shift gamma
            random_gamma = tf.random_uniform([], 0.8, 1.2)
            im_aug  = im_f  ** random_gamma

            # randomly shift brightness
            random_brightness = tf.random_uniform([], 0.5, 2.0)
            im_aug  =  im_aug * random_brightness

            # randomly shift color
            random_colors = tf.random_uniform([in_c], 0.8, 1.2)
            white = tf.ones([batch_size, in_h, in_w])
            color_image = tf.stack([white * random_colors[i] for i in range(in_c)], axis=3)
            im_aug  *= color_image

            # saturate
            im_aug  = tf.clip_by_value(im_aug,  0, 1)

            im_aug = tf.image.convert_image_dtype(im_aug, tf.uint8)

            return im_aug
        #im, intrinsics = random_scaling(im, intrinsics)
        #im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
        im = tf.cast(im, dtype=tf.uint8)
        do_augment  = tf.random_uniform([], 0, 1)
        im = tf.cond(do_augment > 0.5, lambda: random_coloring(im), lambda: im)
        return im, intrinsics

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
        # Assuming the center image is the target frame
        tgt_start_idx = int(img_width * (num_source//2))
        tgt_image = tf.slice(image_seq, 
                             [0, tgt_start_idx, 0], 
                             [-1, img_width, -1])
        # Source frames before the target frame
        src_image_1 = tf.slice(image_seq, 
                               [0, 0, 0], 
                               [-1, int(img_width * (num_source//2)), -1])
        # Source frames after the target frame
        src_image_2 = tf.slice(image_seq, 
                               [0, int(tgt_start_idx + img_width), 0], 
                               [-1, int(img_width * (num_source//2)), -1])
        src_image_seq = tf.concat([src_image_1, src_image_2], axis=1)
        # Stack source frames along the color channels (i.e. [H, W, N*3])
        src_image_stack = tf.concat([tf.slice(src_image_seq, 
                                    [0, i*img_width, 0], 
                                    [-1, img_width, -1]) 
                                    for i in range(num_source)], axis=2)
        src_image_stack.set_shape([img_height, 
                                   img_width, 
                                   num_source * 3])
        tgt_image.set_shape([img_height, img_width, 3])
        return tgt_image, src_image_stack

    def get_multi_scale_intrinsics(self, intrinsics, num_scales):
        intrinsics_mscale = []
        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)
            fy = intrinsics[:,1,1]/(2 ** s)
            cx = intrinsics[:,0,2]/(2 ** s)
            cy = intrinsics[:,1,2]/(2 ** s)
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale
