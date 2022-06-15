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
Data loader with data augmentation.
Only used for training.
"""
from __future__ import absolute_import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from glob import glob

import tensorflow as tf

from tf_smpl.batch_lbs import batch_rodrigues
from util import data_utils
import os

_3D_DATASETS = ['h36m', 'up', 'mpi_inf_3dhp']


def num_examples(datasets):
    _NUM_TRAIN = {
        'lsp': 1000,
        'lsp_ext': 10000,
        'mpii': 18793,
        'h36m': 312188,
        'coco': 80447,
        'mpi_inf_3dhp': 147730,  # without S8
        # Below is number for MOSH/mocap:
        'H3.6': 1559985,  # without S9 and S11
        'CMU': 3934267,
        'jointLim': 181968,
    }

    if not isinstance(datasets, list):
        datasets = [datasets]
    total = 0

    use_dict = _NUM_TRAIN

    for d in datasets:
        total += use_dict[d]
    return total


class DataLoader(object):
    def __init__(self, config):
        self.config = config

        self.use_3d_label = config.use_3d_label

        self.dataset_dir = config.data_dir
        self.datasets = config.datasets
        self.mocap_datasets = config.mocap_datasets
        self.batch_size = config.batch_size
        self.data_format = config.data_format
        self.output_size = config.img_size
        # Jitter params:
        self.trans_max = config.trans_max
        self.scale_range = [config.scale_min, config.scale_max]

        self.image_normalizing_fn = data_utils.rescale_image
        self.rank_size = int(os.getenv('RANK_SIZE'))
        self.rank_id = int(os.getenv('RANK_ID'))

    def load(self):
        if self.use_3d_label:
            image_loader = self.get_loader_w3d_v2()
        else:
            image_loader = self.get_loader()

        return image_loader

    def get_loader(self):
        """
        Outputs:
          image_batch: batched images as per data_format
          label_batch: batched keypoint labels N x K x 3
        """
        files = data_utils.get_all_files(self.dataset_dir, self.datasets)

        do_shuffle = True
        fqueue = tf.train.string_input_producer(
            files, shuffle=do_shuffle, name="input")
        image, label = self.read_data(fqueue, has_3d=False)
        min_after_dequeue = 5000
        num_threads = 8
        capacity = min_after_dequeue + 3 * self.batch_size

        pack_these = [image, label]
        pack_name = ['image', 'label']

        all_batched = tf.train.shuffle_batch(
            pack_these,
            batch_size=self.batch_size,
            num_threads=num_threads,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=False,
            name='input_batch_train')
        batch_dict = {}
        for name, batch in zip(pack_name, all_batched):
            batch_dict[name] = batch

        return batch_dict

    def get_loader_w3d(self):
        """
        Similar to get_loader, but outputs are:
          image_batch: batched images as per data_format
          label_batch: batched keypoint labels N x K x 3
          label3d_batch: batched keypoint labels N x (216 + 10 + 42)
                         216=24*3*3 pose, 10 shape, 42=14*3 3D joints
                         (3D datasets only have 14 joints annotated)
          has_gt3d_batch: batched indicator for
                          existence of [3D joints, 3D SMPL] labels N x 2 - bool
                          Note 3D SMPL is only available for H3.6M.


        Problem is that those datasets without pose/shape do not have them
        in the tfrecords. There's no way to check for this in TF,
        so, instead make 2 string_input_producers, one for data without 3d
        and other for data with 3d.
        And send [2 x *] to train.*batch
        """
        datasets_no3d = [d for d in self.datasets if d not in _3D_DATASETS]
        datasets_yes3d = [d for d in self.datasets if d in _3D_DATASETS]

        files_no3d = data_utils.get_all_files(self.dataset_dir, datasets_no3d)
        files_yes3d = data_utils.get_all_files(self.dataset_dir,
                                               datasets_yes3d)

        # Make sure we have dataset with 3D.
        if len(files_yes3d) == 0:
            print("Dont run this without any datasets with gt 3d")
            import ipdb; ipdb.set_trace()

        do_shuffle = True

        fqueue_yes3d = tf.train.string_input_producer(
            files_yes3d, shuffle=do_shuffle, name="input_w3d")
        image, label, label3d, has_smpl3d = self.read_data(
            fqueue_yes3d, has_3d=True)

        if len(files_no3d) != 0:
            fqueue_no3d = tf.train.string_input_producer(
                files_no3d, shuffle=do_shuffle, name="input_wout3d")
            image_no3d, label_no3d = self.read_data(fqueue_no3d, has_3d=False)
            label3d_no3d = tf.zeros_like(label3d)
            image = tf.parallel_stack([image, image_no3d])
            label = tf.parallel_stack([label, label_no3d])
            label3d = tf.parallel_stack([label3d, label3d_no3d])
            # 3D joint is always available for data with 3d.
            has_3d_joints = tf.constant([True, False], dtype=tf.bool)
            has_3d_smpl = tf.concat([has_smpl3d, [False]], axis=0)
        else:
            # If no "no3d" images, need to make them 1 x *
            image = tf.expand_dims(image, 0)
            label = tf.expand_dims(label, 0)
            label3d = tf.expand_dims(label3d, 0)
            has_3d_joints = tf.constant([True], dtype=tf.bool)
            has_3d_smpl = has_smpl3d

        # Combine 3D bools.
        # each is 2 x 1, column is [3d_joints, 3d_smpl]
        has_3dgt = tf.stack([has_3d_joints, has_3d_smpl], axis=1)

        min_after_dequeue = 2000
        capacity = min_after_dequeue + 3 * self.batch_size

        image_batch, label_batch, label3d_batch, bool_batch = tf.train.shuffle_batch(
            [image, label, label3d, has_3dgt],
            batch_size=self.batch_size,
            num_threads=8,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            enqueue_many=True,
            name='input_batch_train_3d')

        if self.data_format == 'NCHW':
            image_batch = tf.transpose(image_batch, [0, 3, 1, 2])
        elif self.data_format == 'NHWC':
            pass
        else:
            raise Exception("[!] Unkown data_format: {}".format(
                self.data_format))

        batch_dict = {
            'image': image_batch,
            'label': label_batch,
            'label3d': label3d_batch,
            'has3d': bool_batch,
        }

        return batch_dict

    def get_smpl_loader(self):
        """
        Loads dataset in form of queue, loads shape/pose of smpl.
        returns a batch of pose & shape
        """

        data_dirs = [
            join(self.dataset_dir, 'mocap_neutrMosh',
                 'neutrSMPL_%s_*.tfrecord' % dataset)
            for dataset in self.mocap_datasets
        ]
        files = []
        for data_dir in data_dirs:
            files += glob(data_dir)

        if len(files) == 0:
            print('Couldnt find any files!!')
            import ipdb
            ipdb.set_trace()

        return self.get_smpl_loader_from_files_v2(files)

    def get_smpl_loader_from_files(self, files):
        """
        files = list of tf records.
        """
        with tf.name_scope('input_smpl_loader'):
            filename_queue = tf.train.string_input_producer(
                files, shuffle=True)

            mosh_batch_size = self.batch_size * self.config.num_stage

            min_after_dequeue = 1000
            capacity = min_after_dequeue + 3 * mosh_batch_size

            pose, shape = data_utils.read_smpl_data(filename_queue)
            pose_batch, shape_batch = tf.train.batch(
                [pose, shape],
                batch_size=mosh_batch_size,
                num_threads=4,
                capacity=capacity,
                name='input_smpl_batch')

            return pose_batch, shape_batch

    def read_data(self, filename_queue, has_3d=False):
        with tf.name_scope(None, 'read_data', [filename_queue]):
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)
            if has_3d:
                image, image_size, label, center, fname, pose, shape, gt3d, has_smpl3d = data_utils.parse_example_proto(
                    example_serialized, has_3d=has_3d)
                # Need to send pose bc image can get flipped.
                image, label, pose, gt3d = self.image_preprocessing(
                    image, image_size, label, center, pose=pose, gt3d=gt3d)

                # Convert pose to rotation.
                # Do not ignore the global!!
                rotations = batch_rodrigues(tf.reshape(pose, [-1, 3]))
                gt3d_flat = tf.reshape(gt3d, [-1])
                # Label 3d is:
                #   [rotations, shape-beta, 3Djoints]
                #   [216=24*3*3, 10, 42=14*3]
                label3d = tf.concat(
                    [tf.reshape(rotations, [-1]), shape, gt3d_flat], 0)
            else:
                image, image_size, label, center, fname = data_utils.parse_example_proto(
                    example_serialized)
                image, label = self.image_preprocessing(
                    image, image_size, label, center)

            # label should be K x 3
            label = tf.transpose(label)

            if has_3d:
                return image, label, label3d, has_smpl3d
            else:
                return image, label

    def image_preprocessing(self,
                            image,
                            image_size,
                            label,
                            center,
                            pose=None,
                            gt3d=None):
        margin = tf.to_int32(self.output_size / 2)
        with tf.name_scope(None, 'image_preprocessing',
                           [image, image_size, label, center]):
            visibility = label[2, :]
            keypoints = label[:2, :]

            # Randomly shift center.
            print('Using translation jitter: %d' % self.trans_max)
            center = data_utils.jitter_center(center, self.trans_max)
            # randomly scale image.
            image, keypoints, center = data_utils.jitter_scale(
                image, image_size, keypoints, center, self.scale_range)

            # Pad image with safe margin.
            # Extra 50 for safety.
            margin_safe = margin + self.trans_max + 50
            image_pad = data_utils.pad_image_edge(image, margin_safe)
            center_pad = center + margin_safe
            keypoints_pad = keypoints + tf.to_float(margin_safe)

            start_pt = center_pad - margin

            # Crop image pad.
            start_pt = tf.squeeze(start_pt)
            bbox_begin = tf.stack([start_pt[1], start_pt[0], 0])
            bbox_size = tf.stack([self.output_size, self.output_size, 3])

            crop = tf.slice(image_pad, bbox_begin, bbox_size)
            x_crop = keypoints_pad[0, :] - tf.to_float(start_pt[0])
            y_crop = keypoints_pad[1, :] - tf.to_float(start_pt[1])

            crop_kp = tf.stack([x_crop, y_crop, visibility])

            if pose is not None:
                crop, crop_kp, new_pose, new_gt3d = data_utils.random_flip(
                    crop, crop_kp, pose, gt3d)
            else:
                crop, crop_kp = data_utils.random_flip(crop, crop_kp)

            # Normalize kp output to [-1, 1]
            final_vis = tf.cast(crop_kp[2, :] > 0, tf.float32)
            final_label = tf.stack([
                2.0 * (crop_kp[0, :] / self.output_size) - 1.0,
                2.0 * (crop_kp[1, :] / self.output_size) - 1.0, final_vis
            ])
            # Preserving non_vis to be 0.
            final_label = final_vis * final_label

            # rescale image from [0, 1] to [-1, 1]
            crop = self.image_normalizing_fn(crop)
            if pose is not None:
                return crop, final_label, new_pose, new_gt3d
            else:
                return crop, final_label
    
    def parse_w3d(self, record):
        image, image_size, label, center, fname, pose, shape, gt3d, has_smpl3d = data_utils.parse_example_proto(
                record, has_3d=True)
        # Need to send pose bc image can get flipped.
        image, label, pose, gt3d = self.image_preprocessing(
            image, image_size, label, center, pose=pose, gt3d=gt3d)

        # Convert pose to rotation.
        # Do not ignore the global!!
        rotations = batch_rodrigues(tf.reshape(pose, [-1, 3]))
        gt3d_flat = tf.reshape(gt3d, [-1])
        # Label 3d is:
        #   [rotations, shape-beta, 3Djoints]
        #   [216=24*3*3, 10, 42=14*3]
        label3d = tf.concat(
            [tf.reshape(rotations, [-1]), shape, gt3d_flat], 0)
        # label should be K x 3
        label = tf.transpose(label)
        return image, label, label3d, has_smpl3d
    
    def parse_wout3d(self, record):
        image, image_size, label, center, fname = data_utils.parse_example_proto(
                record)
        image, label = self.image_preprocessing(
            image, image_size, label, center)
        # label should be K x 3
        label = tf.transpose(label)
        return image, label
    
    def get_loader_w3d_v1(self):
        datasets_no3d = [d for d in self.datasets if d not in _3D_DATASETS]
        datasets_yes3d = [d for d in self.datasets if d in _3D_DATASETS]

        files_no3d = data_utils.get_all_files(self.dataset_dir, datasets_no3d)
        files_yes3d = data_utils.get_all_files(self.dataset_dir, datasets_yes3d)

        if len(files_yes3d) == 0:
            print("Dont run this without any datasets with gt 3d")
            import ipdb; ipdb.set_trace()
        ds_yes3d = tf.data.TFRecordDataset(files_yes3d)
        ds_yes3d = ds_yes3d.map(self.parse_w3d)
        ds_yes3d = ds_yes3d.shuffle(buffer_size=1000)
        ds_yes3d = ds_yes3d.batch(self.batch_size // 2, drop_remainder=True)
        ds_yes3d = ds_yes3d.repeat()
        iterator_yes3d = ds_yes3d.make_one_shot_iterator()
        image, label, label3d, has_smpl3d = iterator_yes3d.get_next()

        if len(files_no3d) != 0:
            ds_no3d = tf.data.TFRecordDataset(files_no3d)
            ds_no3d = ds_no3d.map(self.parse_wout3d)
            ds_no3d = ds_no3d.shuffle(buffer_size=1000)
            ds_no3d = ds_no3d.batch(self.batch_size // 2, drop_remainder=True)
            ds_no3d = ds_no3d.repeat()
            iterator_no3d = ds_no3d.make_one_shot_iterator()
            image_no3d, label_no3d = iterator_no3d.get_next()
            label3d_no3d = tf.zeros_like(label3d)
            image = tf.concat([image, image_no3d], axis=0)
            label = tf.concat([label, label_no3d], axis=0)
            label3d = tf.concat([label3d, label3d_no3d], axis=0)
            # 3D joint is always available for data with 3d.
            has_3d_joints_yes3d = tf.fill([self.batch_size // 2, 1], True)
            has_3d_joints_no3d = tf.fill([self.batch_size // 2, 1], False)
            has_3d_joints = tf.concat([has_3d_joints_yes3d, has_3d_joints_no3d], axis=0)
            has_3d_smpl = tf.concat([has_smpl3d, tf.fill([self.batch_size // 2, 1], False)], axis=0)
            has_3dgt = tf.concat([has_3d_joints, has_3d_smpl], axis=1)
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])
        elif self.data_format == 'NHWC':
            pass
        else:
            raise Exception("[!] Unkown data_format: {}".format(
                self.data_format))
        batch_dict = {
            'image': image,
            'label': label,
            'label3d': label3d,
            'has3d': has_3dgt,
        }
        return batch_dict
    
    def get_loader_w3d_v2(self):
        datasets_no3d = [d for d in self.datasets if d not in _3D_DATASETS]
        datasets_yes3d = [d for d in self.datasets if d in _3D_DATASETS]

        files_no3d = data_utils.get_all_files(self.dataset_dir, datasets_no3d)
        files_yes3d = data_utils.get_all_files(self.dataset_dir, datasets_yes3d)


        if len(files_yes3d) == 0:
            print("Dont run this without any datasets with gt 3d")
            import ipdb; ipdb.set_trace()
            exit(1)
        
        ds_yes3d = tf.data.Dataset.from_tensor_slices(files_yes3d)
        ds_yes3d = ds_yes3d.shuffle(1024)
        ds_yes3d = ds_yes3d.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=1,
            num_parallel_calls = tf.data.experimental.AUTOTUNE)
        if self.rank_size > 1 :
            ds_yes3d = ds_yes3d.shard(self.rank_size, self.rank_id)
        options = tf.data.Options()
        options.experimental_threading.max_intra_op_parallelism = 1
        ds_yes3d = ds_yes3d.with_options(options)
        ds_yes3d = ds_yes3d.prefetch(buffer_size = self.batch_size)
        ds_yes3d = ds_yes3d.shuffle(buffer_size = 10000)
        ds_yes3d = ds_yes3d.repeat()
        ds_yes3d = ds_yes3d.map(self.parse_w3d, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_yes3d = ds_yes3d.batch(self.batch_size // 2, drop_remainder=True)
        ds_yes3d = ds_yes3d.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        iterator_yes3d = ds_yes3d.make_one_shot_iterator()
        image, label, label3d, has_smpl3d = iterator_yes3d.get_next()

        if len(files_no3d) != 0:
            ds_no3d = tf.data.Dataset.from_tensor_slices(files_no3d)
            ds_no3d = ds_no3d.shuffle(1024)
            ds_no3d = ds_no3d.interleave(
                tf.data.TFRecordDataset,
                cycle_length=10,
                block_length=1,
                num_parallel_calls = tf.data.experimental.AUTOTUNE)
            if self.rank_size > 1:
                ds_no3d = ds_no3d.shard(self.rank_size, self.rank_id)
            options = tf.data.Options()
            options.experimental_threading.max_intra_op_parallelism = 1
            ds_no3d = ds_no3d.with_options(options)
            ds_no3d = ds_no3d.prefetch(buffer_size = self.batch_size)
            ds_no3d = ds_no3d.shuffle(buffer_size = 10000)
            ds_no3d = ds_no3d.repeat()
            ds_no3d = ds_no3d.map(self.parse_wout3d, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds_no3d = ds_no3d.batch(self.batch_size // 2, drop_remainder=True)
            ds_no3d = ds_no3d.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
            iterator_no3d = ds_no3d.make_one_shot_iterator()
            image_no3d, label_no3d = iterator_no3d.get_next()
            label3d_no3d = tf.zeros_like(label3d)
            image = tf.concat([image, image_no3d], axis=0)
            label = tf.concat([label, label_no3d], axis=0)
            label3d = tf.concat([label3d, label3d_no3d], axis=0)
            # 3D joint is always available for data with 3d.
            has_3d_joints_yes3d = tf.fill([self.batch_size // 2, 1], True)
            has_3d_joints_no3d = tf.fill([self.batch_size // 2, 1], False)
            has_3d_joints = tf.concat([has_3d_joints_yes3d, has_3d_joints_no3d], axis=0)
            has_3d_smpl = tf.concat([has_smpl3d, tf.fill([self.batch_size // 2, 1], False)], axis=0)
            has_3dgt = tf.concat([has_3d_joints, has_3d_smpl], axis=1)
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])
        elif self.data_format == 'NHWC':
            pass
        else:
            raise Exception("[!] Unkown data_format: {}".format(
                self.data_format))
        batch_dict = {
            'image': image,
            'label': label,
            'label3d': label3d,
            'has3d': has_3dgt,
        }
        return batch_dict

    def parse_smpl(self, record):
        """
        Parses a smpl Example proto.
        It's contents are:
            'pose'  : 72-D float
            'shape' : 10-D float
        """
        feature_map = {
            'pose': tf.FixedLenFeature((72, ), dtype=tf.float32),
            'shape': tf.FixedLenFeature((10, ), dtype=tf.float32)
        }

        features = tf.parse_single_example(record, feature_map)
        pose = tf.cast(features['pose'], dtype=tf.float32)
        shape = tf.cast(features['shape'], dtype=tf.float32)
        return pose, shape
    
    def get_smpl_loader_from_files_v1(self, files):
        """
        files = list of tf records.
        """
        ds_smpl = tf.data.TFRecordDataset(files)
        mosh_batch_size = self.batch_size * self.config.num_stage
        ds_smpl = ds_smpl.map(self.parse_smpl)
        ds_smpl = ds_smpl.shuffle(buffer_size=1000)
        ds_smpl = ds_smpl.batch(mosh_batch_size, drop_remainder=True)
        ds_smpl = ds_smpl.repeat()
        iterator_smpl = ds_smpl.make_one_shot_iterator()
        pose_batch, shape_batch = iterator_smpl.get_next()
        return pose_batch, shape_batch
    
    def get_smpl_loader_from_files_v2(self, files):
        """
        files = list of tf records.
        """
        mosh_batch_size = self.batch_size * self.config.num_stage
        ds_smpl = tf.data.Dataset.from_tensor_slices(files)
        ds_smpl = ds_smpl.shuffle(1024)
        ds_smpl = ds_smpl.interleave(
            tf.data.TFRecordDataset,
            cycle_length=10,
            block_length=1,
            num_parallel_calls = tf.data.experimental.AUTOTUNE)
        if self.rank_size > 1 :
            ds_smpl = ds_smpl.shard(self.rank_size, self.rank_id)
        options = tf.data.Options()
        options.experimental_threading.max_intra_op_parallelism = 1
        ds_smpl = ds_smpl.with_options(options)
        ds_smpl = ds_smpl.prefetch(buffer_size = mosh_batch_size)
        ds_smpl = ds_smpl.shuffle(buffer_size = 10000)
        ds_smpl = ds_smpl.repeat()
        ds_smpl = ds_smpl.map(self.parse_smpl, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_smpl = ds_smpl.batch(mosh_batch_size, drop_remainder=True)
        ds_smpl = ds_smpl.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        iterator_smpl = ds_smpl.make_one_shot_iterator()
        pose_batch, shape_batch = iterator_smpl.get_next()
        return pose_batch, shape_batch