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
# pylint: enable=line-too-long

import io

from absl import flags
from absl import logging

import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
flags.DEFINE_string(
    'image_file_pattern', '/tmp/*.jpeg',
    'The glob that specifies the image file pattern.')
flags.DEFINE_string(
    'output_dir', '/tmp',
    'The output file that includes bins.')

def main(unused_argv):
  del unused_argv

  image_files = tf.gfile.Glob(FLAGS.image_file_pattern)
  for i, image_file in enumerate(image_files):
    print(' - processing image %d...' % i)
    image = Image.open(image_file)
    image = image.convert('RGB')  # needed for images with 4 channels.
    width, height = image.size

    np_image = (np.array(image.getdata())
                .reshape(height, width, 3).astype(np.uint8))
    np_image_input = np.uint8(np_image.reshape(1, height, width, 3))
    np_image_input.tofile(FLAGS.output_dir + "/{0:05d}.bin".format(i))

if __name__ == '__main__':
  flags.mark_flag_as_required('image_file_pattern')
  flags.mark_flag_as_required('output_dir')
  logging.set_verbosity(logging.INFO)
  tf.app.run(main)
