from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class DataSet(object):
  def __init__(self, images_list_path, num_epoch):
    def parse_example(example):
      content = tf.read_file(example)
      image = tf.image.decode_jpeg(content, channels = 3)

      hr_image = tf.image.resize_images(image, [32, 32])
      lr_image = tf.image.resize_images(image, [8, 8])
      hr_image = tf.cast(hr_image, tf.float32)
      lr_image = tf.cast(lr_image, tf.float32)

      return hr_image, lr_image

    dataset = tf.data.TextLineDataset(images_list_path)

    num_example = 1000
    with open(images_list_path, 'r') as f:
      num_example = len(list(f))

    dataset = dataset.map(parse_example).shuffle(buffer_size = num_example).batch(1, drop_remainder=True).repeat(num_epoch)

    iterator = dataset.make_one_shot_iterator()

    try:
      self.hr_images, self.lr_images = iterator.get_next()
    except tf.errors.OutOfRangeError:
      iterator = dataset.make_one_shot_iterator()
      self.hr_images, self.lr_images = iterator.get_next()