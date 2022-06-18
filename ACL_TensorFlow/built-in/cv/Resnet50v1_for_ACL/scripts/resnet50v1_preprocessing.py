# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import os
import sys
import shutil
import numpy as np
from PIL import Image
from tensorflow.python.ops import control_flow_ops

def eval_image(image, height, width, resize_method,
               central_fraction=0.875, scope=None):

  with tf.name_scope('eval_image'):
    if resize_method == 'crop':
      shape = tf.shape(input=image)
      image = tf.cond(pred=tf.less(shape[0], shape[1]),
                      true_fn=lambda: tf.image.resize(image,
                                                     tf.convert_to_tensor(value=[256, 256 * shape[1] / shape[0]],
                                                                          dtype=tf.int32)),
                      false_fn=lambda: tf.image.resize(image,
                                                     tf.convert_to_tensor(value=[256 * shape[0] / shape[1], 256],
                                                                          dtype=tf.int32)))

      shape = tf.shape(input=image)
      y0 = (shape[0] - height) // 2
      x0 = (shape[1] - width) // 2
      distorted_image = tf.image.crop_to_bounding_box(image, y0, x0, height, width)
      distorted_image.set_shape([height, width, 3])
      means = tf.broadcast_to([123.68, 116.78, 103.94], tf.shape(input=distorted_image))
      return distorted_image - means
    else:  # bilinear
      if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      # Crop the central region of the image with an area containing 87.5% of
      # the original image.
      if central_fraction:
        image = tf.image.central_crop(image, central_fraction=central_fraction)

      if height and width:
        # Resize the image to the specified height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize(image, [height, width],
                                         method=tf.image.ResizeMethod.BILINEAR)
        image = tf.squeeze(image, [0])
      image = tf.subtract(image, 0.5)
      image = tf.multiply(image, 2.0)
      return image

def preprocess(src_path, save_path):
    in_files = os.listdir(src_path)
    in_files.sort()
    resize_shape = [224, 224, 3]
    sqz_mean = np.array([123.68, 116.78, 103.94], np.float32)
    img_std = np.array([[0.5*255, 0.5*255, 0.5*255]], np.float32)
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
        os.makedirs(save_path)
    for file in in_files:
        with tf.Session().as_default():
            if not os.path.isdir(file):
                print(file)
                img_buffer = tf.io.gfile.GFile(os.path.join(src_path, file), 'rb').read()
                img = tf.image.decode_jpeg(img_buffer,channels=3,fancy_upscaling=False,dct_method='INTEGER_FAST')
                img = eval_image(      img,
                                       224,
                                       224,
                                       'crop')
                img = img.eval()
                #img = img * img_std
                img = img + sqz_mean
                img = img.astype(np.uint8, copy=False)
                img.tofile(os.path.join(save_path, file.split('.')[0]+".bin"))
                tf.reset_default_graph()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise Exception("usage: python3 xxx.py [src_path] [save_path]")

    src_path = sys.argv[1]
    save_path = sys.argv[2]
    preprocess(src_path, save_path)
