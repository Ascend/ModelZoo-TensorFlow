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

import random
import os
import numpy as np
from PIL import Image
from numpy.random import randint
import tensorflow as tf
#import matplotlib.pyplot as plt
import time 

import os

from transforms import resize, get_center_crop, get_multi_scale_crop, get_random_horizontal_flip, stack_then_normalize


class Video_3D:
    def __init__(self, info_list, dataset_path, img_format='img_{:04d}.jpg'):
        '''
            info_list: [name, path, total_frame, label]
            tag: 'rgb'(default)
            img_format: 'img_{:04d}.jpg'(default)
        '''
        info_decode_list = list() 
        for info in info_list:
            info_decode_list.append(bytes.decode(info))

        # Element in info_decode_list is shown as below:
        # ['LongJump/v_LongJump_g08_c06', '142', '50']
        # 0 - dir_info | 1 - num_frames | 2 - num_class
        
        self.name = info_decode_list[0]
        # print(type(info_decode_list[0]))
        self.path = os.path.join(bytes.decode(dataset_path), info_decode_list[0])

        # initialzie,to ensure the int is int
        if isinstance(info_decode_list[1], int):
            self.total_frame_num = info_decode_list[1]
        else:
            self.total_frame_num = int(info_decode_list[1])
        if isinstance(info_decode_list[2], int):
            self.label = info_decode_list[2]
        else:
            self.label = int(info_decode_list[2])
        # img_format offer the standard name of picture
        # print('\n===> INFO: {} {} {} {}'.format((info_decode_list[0]), info_decode_list[1], info_decode_list[2], img_format))
        self.img_format = img_format


    def get_frames(self, num_segments, is_training, side_length=224, data_augment=None):
        # assert frame_num <= self.total_frame_num
        frames = list()
        if is_training:
            sample_indices = self._sample_indices(num_segments)
        else:
            sample_indices = self._get_val_indices(num_segments)

        # print('===> INFO: Sample Indices: {}'.format(sample_indices))

        # combine all frames
        for seg_ind in sample_indices:
            seg_imgs = self._load_img(seg_ind)
            frames.extend(seg_imgs)
        
        # frames = transform_data(frames, crop_size=side_length, random_crop=data_augment, random_flip=data_augment)
        
        # what is the meaning of is_numpy
        # frames_np = []
        # for i, img in enumerate(frames):
        #     frames_np.append(np.asarray(img))

        label = self.label 

        # return np.stack(frames_np), label
        return frames, label


    def _sample_indices(self, num_segments):
        """

        :param num_segments: int
        :return: list
        """

        average_duration = self.total_frame_num // num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(num_segments)), average_duration) + randint(average_duration, size=num_segments)
        elif self.total_frame_num > num_segments:
            offsets = np.sort(randint(self.total_frame_num, size=num_segments))
        else:
            offsets = np.zeros((num_segments,))
        return offsets + 1  


    def _get_val_indices(self, num_segments):
        if self.total_frame_num > num_segments:
            tick = self.total_frame_num / float(num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        else:
            offsets = np.zeros((num_segments,))
        return offsets + 1


    def _load_img(self, index):
        img_dir = self.path
        img = Image.open(os.path.join(img_dir, self.img_format.format(index))).convert('RGB')
        return [img]

    def __str__(self):
        return 'Video_3D:\nname: {:s}\nframes: {:d}\nlabel: {:d}\nPath: {:s}'.format(
            self.name, self.total_frame_num, self.label, self.path)


def _get_data_label_from_info(train_info_tensor, is_training, num_segments):
    """ Wrapper for `tf.py_func`, get video clip and label from info list."""
    clip_holder, label_holder = tf.py_func(
        process_video, [train_info_tensor, is_training, num_segments], [tf.float32, tf.int64]) 
    return clip_holder, label_holder


def process_video(data_info, is_training, num_segments, data_augment=None):
    """ Get video clip and label from data info list."""
    video = Video_3D(data_info)
    clip_seq, label_seq = video.get_frames(num_segments, is_training=is_training)

    if is_training:
        clip_seq = get_multi_scale_crop(clip_seq, patch_size=224, scales=[1, .875, .75, .66])
        clip_seq = get_random_horizontal_flip(clip_seq)
    else:
        clip_seq = resize(clip_seq, patch_size=256)
        clip_seq = get_center_crop(clip_seq, patch_size=224)

    # normalize_list = [104, 117, 128]
    # clip_seq = stack_then_normalize(clip_seq, normalize_list)
    clip_seq = [np.array(img)[:,:,::-1].astype('float32') for img in clip_seq]
    clip_seq = np.stack(clip_seq)

    return clip_seq, label_seq


_EACH_VIDEO_TEST_SIZE = 250
_PREFETCH_BUFFER_SIZE = 30
_NUM_PARALLEL_CALLS = 1

fig_show = True

_BATCH_SIZE = 1 if fig_show else 6

num_segments = 8

if __name__ == '__main__':
    train_file = 'splits_txt/ucf101_train_split_1_rawframes.txt'

    f = open(train_file)
    train_info = list()
    for line in f.readlines():
        train_info.append(line.strip().split(' '))
        # print(type(line.strip().split(' ')[0]))
    f.close()
    
    # train_info_tensor = tf.constant(train_info)
    train_info_tensor = train_info
    # print(train_info_tensor)

    # 9537 training samples total
    num_train_sample = len(train_info)
    print(num_train_sample)

    train_info_dataset = tf.data.Dataset.from_tensor_slices((train_info_tensor))
    # train_info_dataset = train_info_dataset.shuffle(buffer_size=num_train_sample)

    train_dataset = train_info_dataset.map(lambda x: _get_data_label_from_info(
        x, is_training=True, num_segments=num_segments), num_parallel_calls=_NUM_PARALLEL_CALLS)
    train_dataset = train_dataset.repeat().batch(_BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=_PREFETCH_BUFFER_SIZE) 

    iterator = tf.data.Iterator.from_structure(
    train_dataset.output_types, train_dataset.output_shapes)
    holder = iterator.get_next()
    train_init_op = iterator.make_initializer(train_dataset)

    sess = tf.Session()
    sess.run(train_init_op) 
    for idx in range(5):
        print('===> IDX: {}'.format(idx+1))
        clip_holder, label_holder = sess.run(holder)

        print(clip_holder.shape)
        print(label_holder.shape)

        if fig_show:
            fig, axs = plt.subplots(ncols=num_segments, figsize=(50, 10))
            
            for i in range(num_segments):
                axs[i].imshow(clip_holder[0, i, :, :, :].astype('int64')) 

            fig.suptitle(str(idx))

        
        plt.show()

