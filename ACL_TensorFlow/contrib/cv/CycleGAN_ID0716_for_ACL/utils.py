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
import random


def convert2int(image):
    """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
  """
    return tf.image.convert_image_dtype((image + 1.0) / 2.0, tf.uint8)


def convert2float(image):
    """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
  """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return (image / 127.5) - 1.0


class ImagePool:
    """ History of generated images
      Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
  """

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image

        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image
