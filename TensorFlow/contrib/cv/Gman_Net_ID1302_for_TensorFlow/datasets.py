"""
datasets
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

import tensorflow as tf
import numpy as np
import gman_flags as df
import os
import gman_constant as constant
from PIL import Image as im


class Image(object):
    """
    Image
    """

    def __init__(self, path, key=None, image_tensor=None, image_index=None, height=None, width=None):
        self.path = path
        self.key = key
        self.image_tensor = image_tensor
        self.image_index = image_index
        self.height = height
        self.width = width


def image_list_shuffle(image_list):
    """

    Args:
        image_list:

    Returns:

    """
    np.random.shuffle(image_list)
    return image_list


def image_input_eval(dir_name, file_names, image_list, clear_dict, clear_image):
    """
        :param dir_name: The directory to read the image
        :param file_names: An empty list to save all image file names
        :return: A list used to save all Image objects.A list used to save names
    """
    if not dir_name:
        raise ValueError('Please supply a data_dir')
    file_list = os.listdir(dir_name)
    for image_filename in file_list:
        if os.path.isdir(os.path.join(dir_name, image_filename)):
            image_input(os.path.join(dir_name, image_filename), file_names, image_list, clear_dict, clear_image)
        elif image_filename.endswith(".png") \
                | image_filename.endswith(".jpg") | image_filename.endswith(".bmp") | image_filename.endswith(".jpeg"):
            file_name = os.path.join(dir_name, image_filename)
            current_image = Image(path=file_name)
            current_image.key = id(current_image)
            current_image.image_index = image_filename[0:constant.IMAGE_INDEX_BIT]
            image_list.append(current_image)
            # Put all clear images into dictionary
            if clear_image:
                if len(image_filename) < constant.IMAGE_INDEX_BIT + constant.IMAGE_SUFFIX_MIN_LENGTH:
                    raise RuntimeError("Incorrect image name: " + image_filename)
                clear_dict[current_image.image_index] = current_image
    if not clear_image:
        image_list_shuffle(image_list)
    for image in image_list:
        file_names.append(image.path)
    return file_names, image_list, clear_dict


def image_input(dir_name, file_names, image_list, clear_dict, clear_image):
    """
        :param dir_name: The directory to read the image
        :param file_names: An empty list to save all image file names
        :return: A list used to save all Image objects.A list used to save names
    """
    if not dir_name:
        raise ValueError('Please supply a data_dir')
    file_list = os.listdir(dir_name)
    for image_filename in file_list:
        if os.path.isdir(os.path.join(dir_name, image_filename)):
            image_input(os.path.join(dir_name, image_filename), file_names, image_list, clear_dict, clear_image)
        elif image_filename.endswith(".png") \
                | image_filename.endswith(".jpg") | image_filename.endswith(".bmp") | image_filename.endswith(".jpeg"):
            file_name = os.path.join(dir_name, image_filename)
            current_image = Image(path=file_name)
            hazed_image = im.open(file_name)
            hazed_image = hazed_image.convert("RGB")
            hazed_image_shape = np.shape(hazed_image)
            current_image.height = hazed_image_shape[0]
            current_image.width = hazed_image_shape[1]
            current_image.key = id(current_image)
            current_image.image_index = image_filename[0:constant.IMAGE_INDEX_BIT]
            image_list.append(current_image)
            # Put all clear images into dictionary
            if clear_image:
                if len(image_filename) < constant.IMAGE_INDEX_BIT + constant.IMAGE_SUFFIX_MIN_LENGTH:
                    raise RuntimeError("Incorrect image name: " + image_filename)
                clear_dict[current_image.image_index] = current_image
    if not clear_image:
        image_list_shuffle(image_list)
    for image in image_list:
        file_names.append(image.path)
    return file_names, image_list, clear_dict


def find_corres_clear_image_rgb(image, clear_dict):
    """

    Args:
        image:
        clear_dict:

    Returns:

    """
    clear_image_obj = clear_dict[image.image_index]
    if not tf.gfile.Exists(clear_image_obj.path):
        raise RuntimeError("Fail to load path from dictionary: " + clear_image_obj.path)
    clear_image = im.open(clear_image_obj.path)
    clear_image = clear_image.convert('RGB')
    return clear_image


def find_corres_clear_image(image, clear_dict):
    """

    Args:
        image:
        clear_dict:

    Returns:

    """
    clear_image_obj = clear_dict[image.image_index]
    if not tf.gfile.Exists(clear_image_obj.path):
        raise RuntimeError("Fail to load path from dictionary: " + clear_image_obj.path)
    with tf.gfile.GFile(clear_image_obj.path, 'rb') as fid:
        clear_image = fid.read()
    # clear_image = clear_image.convert('RGB')
    return clear_image


def bytes_feature(value):
    """

    Args:
        value:

    Returns:

    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    """

    Args:
        value:

    Returns:

    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def convert_to_tfrecord(hazed_image_list, dict_name, tfrecord_path, test_image_list):
    """

    Args:
        hazed_image_list:
        dict_name:
        tfrecord_path:
        test_image_list:

    Returns:

    """
    counter = 0
    test_clear_index_list = []
    for image in test_image_list:
        test_clear_index = image.image_index
        test_clear_index_list.append(test_clear_index)
    if len(hazed_image_list) == 0:
        raise RuntimeError("No example found for training! Please check your training data set!")
    for image in hazed_image_list:
        if not tf.gfile.Exists(image.path):
            raise ValueError('Failed to find image from path: ' + image.path)
    print('Start converting data into tfrecords...')
    writer_options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(tfrecord_path, options=writer_options)
    try:
        for image in hazed_image_list:
            try:
                with tf.gfile.GFile(image.path, 'rb') as fid:
                    hazed_image = fid.read()
                # hazed_image_shape = np.shape(hazed_image)
                # haze_height = hazed_image_shape[0]
                # haze_width = hazed_image_shape[1]
                # reshape_hazed_image_arr = np.array(hazed_image)
                # hazed_image_raw = reshape_hazed_image_arr.tostring()
                # ################Getting corresponding clear images#########################
                clear_image = find_corres_clear_image(image, dict_name)
                # reshape_clear_image_arr = np.array(clear_image)
                # clear_image_raw = reshape_clear_image_arr.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'hazed_image_raw': bytes_feature(hazed_image),
                    'clear_image_raw': bytes_feature(clear_image),
                    'hazed_height': int64_feature(image.height),
                    'hazed_width': int64_feature(image.width),
                }))
                writer.write(example.SerializeToString())
                counter += 1
            except IOError as e:
                raise RuntimeError('Could not read:', image.path)
    finally:
        writer.close()
    print('Transform done! Totally transformed ' + str(counter * 2) + ' pairs of examples.')


def parse_record(example):
    """

    Args:
        example:

    Returns:

    """
    img_features = tf.parse_single_example(
        example,
        features={
            'hazed_image_raw': tf.FixedLenFeature([], tf.string),
            'clear_image_raw': tf.FixedLenFeature([], tf.string),
            'hazed_height': tf.FixedLenFeature([], tf.int64),
            'hazed_width': tf.FixedLenFeature([], tf.int64),
        })
    hazed_image = tf.image.decode_image(img_features['hazed_image_raw'], channels=3, dtype=tf.float32,
                                        expand_animations=False)
    hazed_height = tf.cast(img_features['hazed_height'], tf.int32)
    hazed_width = tf.cast(img_features['hazed_width'], tf.int32)
    # hazed_image = tf.reshape(hazed_image, [hazed_height, hazed_width, 3])
    clear_image = tf.image.decode_image(img_features['clear_image_raw'], channels=3, dtype=tf.float32,
                                        expand_animations=False)
    # clear_image = tf.reshape(clear_image, [hazed_height, hazed_width, 3])
    # stack the haze and clear images on channel axis
    composed_images = tf.concat([hazed_image, clear_image], axis=2)
    if hazed_height < df.FLAGS.input_image_height:
        composed_images = tf.image.resize_images(composed_images, [df.FLAGS.input_image_height, hazed_width])
    if hazed_width < df.FLAGS.input_image_width:
        composed_images = tf.image.resize_images(composed_images, [hazed_height, df.FLAGS.input_image_width])
    croped_composed_images = tf.random_crop(composed_images,
                                            [df.FLAGS.input_image_height, df.FLAGS.input_image_width, 6])
    hazed_image = croped_composed_images[:, :, :3]
    clear_image = croped_composed_images[:, :, 3:]
    # hazed_image = tf.image.resize_images(croped_composed_images[:, :, :3],[df.FLAGS.input_image_height, df.FLAGS.input_image_width])
    # clear_image = tf.image.resize_images(croped_composed_images[:, :, 3:],[df.FLAGS.input_image_height, df.FLAGS.input_image_width])
    if df.FLAGS.use_fp16:
        hazed_image = tf.image.convert_image_dtype(hazed_image, tf.float16)
        clear_image = tf.image.convert_image_dtype(clear_image, tf.float16)
    else:
        hazed_image = tf.image.convert_image_dtype(hazed_image, tf.float32)
        clear_image = tf.image.convert_image_dtype(clear_image, tf.float32)
    return hazed_image, clear_image


def read_input():
    """

    Returns:

    """
    batch_size = 8
    with tf.device('/cpu:0'):
        dataset = tf.data.TFRecordDataset(["./Test_record/test.tfrecords"], compression_type='ZLIB').repeat()
        dataset = dataset.map(parse_record, num_parallel_calls=64)
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(64)
        iterator = dataset.make_initializable_iterator()
        # train_batch_element = iterator.get_next()
        return iterator


if __name__ == '__main__':
    pass
    # iterator = read_input()
    # for i in range(280000):
    #     hazed_image, clear_image=iterator.get_next()
    #     print(hazed_image.shape)
    #     if int(hazed_image.shape[1]<224):
    #         print('ok')
    # Frames used to save clear training image information
    # _clear_train_file_names = []
    # _clear_train_img_list = []
    # _clear_train_directory = {}
    # _hazed_train_file_names = []
    # _hazed_train_img_list = []
    # _clear_test_img_list = []
    #
    # image_input('./ClearImages/TestImages', _clear_train_file_names, _clear_train_img_list,
    #                 _clear_train_directory, clear_image=True)
    # if len(_clear_train_img_list) == 0:
    #     raise RuntimeError("No image found! Please supply clear images for training or eval ")
    # # Hazed training image pre-process
    # image_input('./HazeImages/TestImages', _hazed_train_file_names, _hazed_train_img_list,
    #                 clear_dict=None, clear_image=False)
    # if len(_hazed_train_img_list) == 0:
    #     raise RuntimeError("No image found! Please supply hazed images for training or eval ")
    #
    # convert_to_tfrecord(_hazed_train_img_list, _clear_train_directory,
    #                             './Test_record/test.tfrecords', _clear_test_img_list)

    # image_input('./ClearImages/TrainImages', _clear_train_file_names, _clear_train_img_list,
    #                 _clear_train_directory, clear_image=True)
    # if len(_clear_train_img_list) == 0:
    #     raise RuntimeError("No image found! Please supply clear images for training or eval ")
    # # Hazed training image pre-process
    # image_input('./HazeImages/TrainImages', _hazed_train_file_names, _hazed_train_img_list,
    #                 clear_dict=None, clear_image=False)
    # if len(_hazed_train_img_list) == 0:
    #     raise RuntimeError("No image found! Please supply hazed images for training or eval ")

    # convert_to_tfrecord(_hazed_train_img_list, _clear_train_directory,
    #                             './Train_record/train.tfrecords', _clear_test_img_list)
