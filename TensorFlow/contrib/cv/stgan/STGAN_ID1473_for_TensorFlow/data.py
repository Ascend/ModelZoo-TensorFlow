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
os.environ["ASCEND_DEVICE_ID"] = "6"

import numpy as np
import tensorflow as tf
from tflib.utils import session
import random


def batch_dataset(dataset, batch_size, prefetch_batch=2, drop_remainder=True, filter=None,
                  map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1):
    if filter:
        dataset = dataset.filter(filter)

    if map_func:
        dataset = dataset.map(map_func, num_parallel_calls=num_threads)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.repeat(repeat)

    dataset = dataset.prefetch(prefetch_batch)

    return dataset


def disk_image_batch_dataset(img_paths, batch_size, labels=None, prefetch_batch=2, drop_remainder=True, filter=None,
                             map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1):
    """Disk image batch dataset.

    This function is suitable for jpg and png files

    img_paths: string list or 1-D tensor, each of which is an iamge path
    labels: label list/tuple_of_list or tensor/tuple_of_tensor, each of which is a corresponding label
    """
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    elif isinstance(labels, tuple):
        dataset = tf.data.Dataset.from_tensor_slices((img_paths,) + tuple(labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    def parse_func(path, *label):
        img = tf.read_file(path)
        img = tf.image.decode_png(img, 3)
        return (img,) + label

    if map_func:
        def map_func_(*args):
            return map_func(*parse_func(*args))
    else:
        map_func_ = parse_func

    # dataset = dataset.map(parse_func, num_parallel_calls=num_threads) is slower

    dataset = batch_dataset(dataset, batch_size, prefetch_batch, drop_remainder, filter,
                            map_func_, num_threads, shuffle, buffer_size, repeat)

    return dataset


class Dataset(object):

    def __init__(self,args):
        self.train_image_num=args.train_image_num  
        self._dataset = None
        self._iterator = None
        self._batch_op = None
        self._sess = None

    def __del__(self):
        if self._sess:
            self._sess.close()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            b = self.get_next()
        except:
            raise StopIteration
        else:
            return b

    next = __next__

    def get_next(self):
        return self._sess.run(self._batch_op)

    def reset(self, feed_dict={}):
        self._sess.run(self._iterator.initializer, feed_dict=feed_dict)

    def _bulid(self, dataset, sess=None):
        self._dataset = dataset

        self._iterator = dataset.make_initializable_iterator()
        self._batch_op = self._iterator.get_next()
        if sess:
            self._sess = sess
        else:
            self._sess = session()

        try:
            self.reset()
        except:
            pass

    @property
    def dataset(self):
        return self._dataset

    @property
    def iterator(self):
        return self._iterator

    @property
    def batch_op(self):
        return self._batch_op


class Celeba(Dataset):

    att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
                'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
                'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
                'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
                'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
                'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
                'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
                'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
                'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
                'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
                'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
                'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
                'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

    def __init__(self, args, data_dir, atts, img_resize, batch_size, prefetch_batch=2, drop_remainder=True,
                 num_threads=16, shuffle=True, buffer_size=4096, repeat=-1, sess=None, part='train', crop=True, im_no=None):
        self.train_image_num=args.train_image_num
        super(Celeba, self).__init__(args)

        list_file = os.path.join(data_dir, 'list_attr_celeba.txt')
        if crop:
            img_dir_jpg = os.path.join(data_dir, 'img_align_celeba')
            img_dir_png = os.path.join(data_dir, 'img_align_celeba_png')
        else:
            img_dir_jpg = os.path.join(data_dir, 'img_crop_celeba')
            img_dir_png = os.path.join(data_dir, 'img_crop_celeba_png')

        names = np.loadtxt(list_file, skiprows=2, usecols=[0], dtype=np.str)
        if os.path.exists(img_dir_png):
            img_paths = [os.path.join(img_dir_png, name.replace('jpg', 'png')) for name in names]
        elif os.path.exists(img_dir_jpg):
            img_paths = [os.path.join(img_dir_jpg, name) for name in names]

        att_id = [Celeba.att_dict[att] + 1 for att in atts]
        labels = np.loadtxt(list_file, skiprows=2, usecols=att_id, dtype=np.int64)

        if img_resize == 64:
            # crop as how VAE/GAN do
            offset_h = 40
            offset_w = 15
            img_size = 148
        else:
            offset_h = 26
            offset_w = 3
            img_size = 170

        def _map_func(img, label):
            if crop:
                img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, img_size, img_size)
            # img = tf.image.resize_images(img, [img_resize, img_resize]) / 127.5 - 1
            # or
            img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            label = (label + 1) // 2
            return img, label

        if im_no is not None:
            drop_remainder = False
            shuffle = False
            repeat = 1
            img_paths = [img_paths[i-1] for i in im_no]
            labels = labels[[i-1 for i in im_no]]
        elif part == 'test':
            drop_remainder = False
            shuffle = False
            repeat = 1
            img_paths = img_paths[182637:]
            labels = labels[182637:]
        elif part == 'val':
            img_paths = img_paths[182000:182637]
            labels = labels[182000:182637]
        else:
            #img_paths = img_paths[:182000]
            #labels = labels[:182000]
            img_paths = img_paths[:self.train_image_num]  # CI训练使用1000张图
            labels = labels[:self.train_image_num]    # CI训练使用1000张图

        dataset = disk_image_batch_dataset(img_paths=img_paths,
                                           labels=labels,
                                           batch_size=batch_size,
                                           prefetch_batch=prefetch_batch,
                                           drop_remainder=drop_remainder,
                                           map_func=_map_func,
                                           num_threads=num_threads,
                                           shuffle=shuffle,
                                           buffer_size=buffer_size,
                                           repeat=repeat)
        self._bulid(dataset, sess)

        self._img_num = len(img_paths)

    def __len__(self):
        return self._img_num

    @staticmethod
    def check_attribute_conflict(att_batch, att_name, att_names):
        def _set(att, value, att_name):
            if att_name in att_names:
                att[att_names.index(att_name)] = value

        att_id = att_names.index(att_name)

        for att in att_batch:
            if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] == 1:
                _set(att, 0, 'Bangs')
            elif att_name == 'Bangs' and att[att_id] == 1:
                _set(att, 0, 'Bald')
                _set(att, 0, 'Receding_Hairline')
            elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] == 1:
                for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    if n != att_name:
                        _set(att, 0, n)
                #_set(att, 0, 'bald')
            elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] == 1:
                for n in ['Straight_Hair', 'Wavy_Hair']:
                    if n != att_name:
                        _set(att, 0, n)
# Removed since `Mustache` and `No_Beard` are not conflict.
# But the two attributes are not well labeled in the dataset.
#            elif att_name in ['Mustache', 'No_Beard'] and att[att_id] == 1:
#                for n in ['Mustache', 'No_Beard']:
#                    if n != att_name:
#                        _set(att, 0, n)

        return att_batch
    
    @staticmethod
    def check_random_attribute_conflict(att_batch, att_names, hair_color=None):
        """ For randomly generated attributes, tested but not used in this repo. """
        def _set(att, value, att_name):
            if att_name in att_names:
                att[att_names.index(att_name)] = value
        
        def _idx(att_name):
            if att_name in att_names:
                return att_names.index(att_name)
            return None
        
        for att in att_batch:
            valid_atts = [i for i in ['Receding_Hairline', 'Bald'] if i in att_names]
            if 'Bangs' in att_names and att[_idx('Bangs')] == 1 \
               and len(valid_atts) > 0 and sum([att[_idx(i)] for i in valid_atts]) > 0:
                    _set(att, 0, 'Bangs') if random.random() < 0.5 else [_set(att, 0, i) for i in valid_atts]
#            hair_color = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
            if hair_color is not None and sum([att[_idx(i)] for i in hair_color]) > 1:
                one = random.randint(0, len(hair_color))
                for i in range(len(hair_color)):
                    _set(att, 1 if i==one else 0, hair_color[i])
            if 'Straight_Hair' in att_names and 'Wavy_Hair' in att_names and att[_idx('Straight_Hair')] == 1 and att[_idx('Wavy_Hair')] == 1:
                _set(att, 0, 'Straight_Hair') if random.random() < 0.5 else _set(att, 0, 'Wavy_Hair')
#            if 'Mustache' in att_names and 'No_Beard' in att_names and att[_idx('Mustache')] == 1 and att[_idx('No_Beard')] == 1:
#                _set(att, 0, 'Mustache') if random.random() < 0.5 else _set(att, 0, 'No_Beard')
        return att_batch

if __name__ == '__main__':
    import imlib as im
    atts = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
    data = Celeba('/home/test_user02/yuxh/celeba/', atts, 128, 1, part='test')
    batch = data.get_next()
    print(len(data))
    for idx, batch in enumerate(data):
        print(idx + 182638)
    #print(batch[1][1], batch[1].dtype)
    #print(batch[0].min(), batch[1].max(), batch[0].dtype)
    #im.imshow(batch[0][1])
    #im.show()
