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
import os

import numpy as np
import tensorflow as tf

from dataloaders.base_loader import BaseLoader
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_input_path', 'LR',
                       'Base path of the input images. For example, if you specify this argument to \'LR\', the downscaled images by a factor of 4 should be in \'LR/x4/\'.')
tf.flags.DEFINE_string('data_truth_path', 'HR',
                       'Base path of the ground-truth images. The image name should be the same as that of the input (downscaled) image.')
tf.flags.DEFINE_bool('data_cached', False, 'If true, cache the data on the memory.')

if FLAGS.chip == 'npu':
    from npu_bridge.npu_init import *


def create_loader():
    return BasicLoader()


class BasicLoader(BaseLoader):
    def __init__(self):
        super().__init__()

    def prepare(self):
        self.scale_list = list(map(lambda x: int(x), FLAGS.scales.split(',')))

        # retrieve image name list
        # 为什么只取第一项？
        scale = self.scale_list[0]
        input_path = os.path.join(FLAGS.data_input_path, ('x%d' % (scale)))
        # self.image_name_list = [f for f in os.listdir(input_path) if f.lower().endswith('.png')]
        self.image_name_list = [f for f in os.listdir(input_path)]
        tf.logging.info('data: %d images are prepared (%s)' % (
        len(self.image_name_list), 'caching enabled' if FLAGS.data_cached else 'caching disabled'))

        # initialize image reading session
        self.tf_image_graph = tf.Graph()
        with self.tf_image_graph.as_default():
            self.tf_image_path = tf.placeholder(tf.string, [])

            tf_image = tf.read_file(self.tf_image_path)
            tf_image = tf.image.decode_png(tf_image, channels=3, dtype=tf.uint8)

            self.tf_image = tf_image

            init = tf.global_variables_initializer()

            # trans
            config = tf.ConfigProto()
            custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
            custom_op.name = "NpuOptimizer"
            config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
            config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

            self.tf_image_session = tf.Session(config=config)
            self.tf_image_session.run(init)
        # initialize cached list
        self.cached_input_image_list = {}
        for scale in self.scale_list:
            self.cached_input_image_list[scale] = {}
        self.cached_truth_image_list = {}

    def get_num_images(self):
        return len(self.image_name_list)

    def get_patch_batch(self, batch_size, scale, input_patch_size):
        input_list = []
        truth_list = []

        for _ in range(batch_size):
            input_patch, truth_patch = self.get_random_image_patch_pair(scale=scale, input_patch_size=input_patch_size)
            input_list.append(input_patch)
            truth_list.append(truth_patch)

        return input_list, truth_list

    def get_random_image_patch_pair(self, scale, input_patch_size):
        # select an image
        image_index = np.random.randint(self.get_num_images())

        # retrieve image
        input_patch, truth_patch = self.get_image_patch_pair(image_index=image_index, scale=scale,
                                                             input_patch_size=input_patch_size)

        # finalize
        return input_patch, truth_patch

    def get_image_patch_pair(self, image_index, scale, input_patch_size):
        # retrieve image
        input_image, truth_image, _ = self.get_image_pair(image_index=image_index, scale=scale)

        # randomly crop
        truth_patch_size = input_patch_size * scale
        height, width, _ = input_image.shape
        input_x = np.random.randint(width - input_patch_size)
        input_y = np.random.randint(height - input_patch_size)
        truth_x = input_x * scale
        truth_y = input_y * scale
        input_patch = input_image[input_y:(input_y + input_patch_size), input_x:(input_x + input_patch_size), :]
        truth_patch = truth_image[truth_y:(truth_y + truth_patch_size), truth_x:(truth_x + truth_patch_size), :]

        # randomly rotate
        rot90_k = np.random.randint(4) + 1
        input_patch = np.rot90(input_patch, rot90_k)
        truth_patch = np.rot90(truth_patch, rot90_k)

        # randomly flip
        flip = (np.random.uniform() < 0.5)
        if (flip):
            input_patch = np.fliplr(input_patch)
            truth_patch = np.fliplr(truth_patch)

        # finalize
        return input_patch, truth_patch

    def get_image_pair(self, image_index, scale):
        # retrieve image
        image_name = self.image_name_list[image_index]
        input_image = self._get_input_image(scale, image_name)
        truth_image = self._get_truth_image(image_name)

        # finalize
        return input_image, truth_image, image_name

    def _get_input_image(self, scale, image_name):
        image = None
        has_cached = False
        if (FLAGS.data_cached):
            if (image_name in self.cached_input_image_list[scale]):
                image = self.cached_input_image_list[scale][image_name]
                has_cached = True

        if (image is None):
            image_path = os.path.join(FLAGS.data_input_path, ('x%d' % (scale)), image_name)
            image = self.tf_image_session.run(self.tf_image, feed_dict={self.tf_image_path: image_path})

        if (FLAGS.data_cached and (not has_cached)):
            self.cached_input_image_list[scale][image_name] = image

        return image

    def _get_truth_image(self, image_name):
        image = None
        has_cached = False
        if (FLAGS.data_cached):
            if (image_name in self.cached_truth_image_list):
                image = self.cached_truth_image_list[image_name]
                has_cached = True

        if (image is None):
            image_path = os.path.join(FLAGS.data_truth_path, image_name)
            image = self.tf_image_session.run(self.tf_image, feed_dict={self.tf_image_path: image_path})

        if (FLAGS.data_cached and (not has_cached)):
            self.cached_truth_image_list[image_name] = image

        return image
