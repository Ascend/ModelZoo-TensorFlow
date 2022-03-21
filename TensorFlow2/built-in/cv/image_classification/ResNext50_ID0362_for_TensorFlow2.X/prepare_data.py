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

import tensorflow as tf
import pathlib
from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, \
    BATCH_SIZE, train_tfrecord, valid_tfrecord, test_tfrecord
from parse_tfrecord import get_parsed_dataset


def load_and_preprocess_image(image_raw, data_augmentation=False):
    # decode
    image_tensor = tf.io.decode_image(contents=image_raw, channels=CHANNELS, dtype=tf.dtypes.float32)

    if data_augmentation:
        image = tf.image.random_flip_left_right(image=image_tensor)
        image = tf.image.resize_with_crop_or_pad(image=image,
                                                 target_height=int(IMAGE_HEIGHT * 1.2),
                                                 target_width=int(IMAGE_WIDTH * 1.2))
        image = tf.image.random_crop(value=image, size=[IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])
        image = tf.image.random_brightness(image=image, max_delta=0.5)
    else:
        image = tf.image.resize(image_tensor, [IMAGE_HEIGHT, IMAGE_WIDTH])

    return image


def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: {label : index}
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in all_image_path]

    return all_image_path, all_image_label


def get_the_length_of_dataset(dataset):
    count = 0
    for i in dataset:
        count += 1
    return count


def generate_datasets():
    train_dataset = get_parsed_dataset(tfrecord_name=train_tfrecord)
    valid_dataset = get_parsed_dataset(tfrecord_name=valid_tfrecord)
    test_dataset = get_parsed_dataset(tfrecord_name=test_tfrecord)

    train_count = get_the_length_of_dataset(train_dataset)
    valid_count = get_the_length_of_dataset(valid_dataset)
    test_count = get_the_length_of_dataset(test_dataset)

    # read the dataset in the form of batch
    train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=BATCH_SIZE)

    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count
