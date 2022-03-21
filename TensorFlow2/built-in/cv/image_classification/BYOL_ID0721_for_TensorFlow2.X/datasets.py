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
import tensorflow as tf
from augmentation import augment_image_pretraining, augment_image_finetuning

class CIFAR10:

    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()
        self.num_train_images, self.num_test_images = self.y_train.shape[0], self.y_test.shape[0]
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'sheep', 'truck']

        # Normalize training and testing images
        self.x_train = tf.cast(self.x_train / 255., tf.float32)
        self.x_test = tf.cast(self.x_test / 255., tf.float32)

        self.y_train = tf.cast(tf.squeeze(self.y_train), tf.int32)
        self.y_test = tf.cast(tf.squeeze(self.y_test), tf.int32)


    def get_batch_pretraining(self, batch_id, batch_size):
        '''
        get pretraining batch
        :param batch_id:
        :param batch_size:
        :return: x_batch_1, x_batch_2
        '''
        x_batch = tf.slice(self.x_train, [batch_id*batch_size, 0, 0, 0], [batch_size, -1, -1, -1])
        x_batch_1 = tf.map_fn(augment_image_pretraining, x_batch)
        x_batch_2 = tf.map_fn(augment_image_pretraining, x_batch)
        return x_batch_1, x_batch_2  # (bs, 32, 32, 3), (bs, 32, 32, 3)


    def get_batch_finetuning(self, batch_id, batch_size):
        x_batch = tf.slice(self.x_train, [batch_id * batch_size, 0, 0, 0], [batch_size, -1, -1, -1])
        x_batch = tf.map_fn(augment_image_finetuning, x_batch)
        y_batch = tf.slice(self.y_train, [batch_id*batch_size], [batch_size])
        return x_batch, y_batch  # (bs, 32, 32, 3), (bs)


    def get_batch_testing(self, batch_id, batch_size):
        x_batch = tf.slice(self.x_test, [batch_id*batch_size, 0, 0, 0], [batch_size, -1, -1, -1])
        y_batch = tf.slice(self.y_test, [batch_id*batch_size], [batch_size])
        return x_batch, y_batch  # (bs, 32, 32, 3), (bs)


    def shuffle_training_data(self):
        random_ids = tf.random.shuffle(tf.range(self.num_train_images))
        self.x_train = tf.gather(self.x_train, random_ids)
        self.y_train = tf.gather(self.y_train, random_ids)
