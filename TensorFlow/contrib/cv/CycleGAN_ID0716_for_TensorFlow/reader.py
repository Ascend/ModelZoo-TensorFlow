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
import utils


class Reader():
    def __init__(self, tfrecords_file, image_size=256,
                 min_queue_examples=1000, batch_size=1, name=''):
        """
    Args:
      tfrecords_file: string, tfrecords file path
      min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
      batch_size: integer, number of images per batch
      num_threads: integer, number of preprocess threads
    """
        self.tfrecords_file = tfrecords_file
        self.image_size = image_size
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.name = name

    def _preprocess(self, image):
        image = tf.image.resize_images(image, size=(self.image_size, self.image_size))
        image = utils.convert2float(image)
        image.set_shape([self.image_size, self.image_size, 3])
        return image

    def feed(self):
        dataset = tf.data.TFRecordDataset(self.tfrecords_file, buffer_size=256 << 20)
        data_ = dataset.map(self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
            self.min_queue_examples + 3 * self.batch_size).repeat().batch(self.batch_size, drop_remainder=True)
        #data = data_.make_one_shot_iterator()
        #return data.get_next()
        data = data_.make_initializable_iterator()
        return data.get_next(),data.initializer
    def parse_function(self, example_proto):
        dics = {
            'image/file_name': tf.FixedLenFeature([], tf.string),
            'image/encoded_image': tf.FixedLenFeature([], tf.string),
        }
        parsed_example = tf.parse_single_example(example_proto, dics)
        image_buffer = parsed_example['image/encoded_image']
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = self._preprocess(image)
        return image
