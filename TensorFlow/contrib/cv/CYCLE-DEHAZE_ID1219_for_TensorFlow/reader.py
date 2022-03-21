# MIT License

# Copyright (c) 2018 Deniz Engin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
from npu_bridge.npu_init import *
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import utils
import pdb


class Reader():
    def __init__(self, tfrecords_file, image_size1=256, image_size2=256, min_queue_examples=1000, batch_size=1,
                 num_threads=8, name=''):
        """
        Args:
          tfrecords_file: string, tfrecords file path
          min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
          batch_size: integer, number of images per batch
          num_threads: integer, number of preprocess threads
        """
        self.tfrecords_file = tfrecords_file
        self.image_size1 = image_size1
        self.image_size2 = image_size2
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.name = name

    def _preprocess(self, image):
        image = tf.image.resize_images(image, size=(self.image_size1, self.image_size2))
        image = utils.convert2float(image)
        image.set_shape([self.image_size1, self.image_size2, 3])
        return image

    def _parse_record(self, example):
        features = {
            'image/file_name': tf.FixedLenFeature([], tf.string),
            'image/encoded_image': tf.FixedLenFeature([], tf.string),
        }
        parsed_features = tf.parse_single_example(example, features=features)
        image_buffer = parsed_features['image/encoded_image']
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = self._preprocess(image)
        return image

    def feed(self):
        """
        Returns:
          images: 4D tensor [batch_size, image_width, image_height, image_depth]
        """
        with tf.name_scope(self.name):
            dataset = tf.data.TFRecordDataset(self.tfrecords_file).repeat()
            dataset = dataset.map(lambda value: self._parse_record(value), num_parallel_calls=4)
            dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(64)
            iterator = dataset.make_one_shot_iterator()
            images = iterator.get_next()
            tf.summary.image('inputt', images)
        return images


def test_reader():
    TRAIN_FILE_1 = 'tfrecords/clearImage.tfrecords'
    TRAIN_FILE_2 = 'tfrecords/hazyImage.tfrecords'

    with tf.Graph().as_default():
        reader1 = Reader(TRAIN_FILE_1, batch_size=2)
        reader2 = Reader(TRAIN_FILE_2, batch_size=2)
        images_op1 = reader1.feed()
        images_op2 = reader2.feed()
        #pdb.set_trace()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        sess = tf.Session(config=npu_config_proto(config_proto=config))
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while step < 1:
                batch_images1, batch_images2 = sess.run([images_op1, images_op2])
                print("image shape: {}".format(batch_images1))
                print("image shape: {}".format(batch_images2))
                print("=" * 10)
                step += 1
        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    test_reader()
