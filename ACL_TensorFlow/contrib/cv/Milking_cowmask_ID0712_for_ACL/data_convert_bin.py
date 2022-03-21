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
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
from absl import flags,app

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'dataset_path', default='./dataset',
    help=('Directory to store dataset data'))
flags.DEFINE_string(
    'bin_path', default='./bin',
    help=('Directory to store bin data'))
flags.DEFINE_integer(
    'batch_size', default=1,
    help=('Batch size for inference,need to be divided by sum_samples.'))

MEAN_RGB = [0.4914 * 255, 0.4822 * 255, 0.4465 * 255]
STDDEV_RGB = [0.2470 * 255, 0.2435 * 255, 0.2616 * 255]

mean_rgb = tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=tf.float32)
stddev_rgb = tf.constant(STDDEV_RGB, shape=[1, 1, 3],
                             dtype=tf.float32)

def _preprocess_eval_image(image, mean_rgb, stddev_rgb):
  image = tf.cast(image, tf.float32)
  image = (image - mean_rgb) / stddev_rgb
  return image


def _eval_map_fn(x):
    """Pre-process evaluation sample."""
    image = _preprocess_eval_image(x['image'], mean_rgb, stddev_rgb)
    batch = {'image': image, 'label': x['label']}
    return batch

def main(argv):
    if len(argv) > 3:
        raise app.UsageError('Too many command-line arguments.')
    image_path = os.path.join(FLAGS.bin_path,'image_' + str(FLAGS.batch_size))
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    label_path = os.path.join(FLAGS.bin_path, 'label_' + str(FLAGS.batch_size))
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    test_ds = tfds.load('cifar10', split='test', data_dir=FLAGS.dataset_path)
    test_ds = test_ds.map(_eval_map_fn, num_parallel_calls=128)
    test_ds = test_ds.batch(FLAGS.batch_size, drop_remainder=True)
    iterator = test_ds.make_one_shot_iterator()
    image_batch = iterator.get_next()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        i = 1
        while i<= 10000/FLAGS.batch_size:
            images = sess.run(image_batch)

            name_image = ''.join([image_path, '/', str(i), '_image', '.bin'])
            name_label = ''.join([label_path, '/', str(i), '_label', '.txt'])
            image = images['image'].astype(np.float32)
            label = images['label'].astype(np.float32)
            image.tofile(name_image)
            np.savetxt(name_label, label)
            i+=1


if __name__ == '__main__':
    app.run(main)
