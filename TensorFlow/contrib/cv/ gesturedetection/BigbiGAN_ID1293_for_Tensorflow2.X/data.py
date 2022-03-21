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

import logging
import tensorflow_datasets as tfds
import tensorflow as tf

NUM_CALLS = tf.data.experimental.AUTOTUNE
NUM_PREFETCH = tf.data.experimental.AUTOTUNE

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image = image/255.0
    # Rescale image to 32x32 if mnist/fmnist
    image = tf.image.resize(image, [32,32])
    return image, label


def get_dataset(config):
    datasets, ds_info = tfds.load(name=config.dataset, with_info=True, as_supervised=True, data_dir=config.dataset_path)
    train_data, test_data = datasets['train'], datasets['test']
    return train_data, test_data


def get_train_pipeline(dataset,config):
    if(config.cache_dataset):
        dataset = dataset.cache()
    dataset = dataset.shuffle(config.data_buffer_size).repeat().map(scale, num_parallel_calls=NUM_CALLS).batch(config.train_batch_size,drop_remainder=True).prefetch(config.data_buffer_size)
    return dataset

