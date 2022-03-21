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


from npu_bridge.npu_init import *
from abc import abstractmethod

import tensorflow as tf

class BevFeatureExtractor:

    def __init__(self, extractor_config):
        self.config = extractor_config

    def preprocess_input(self, tensor_in, output_shape):
        """Preprocesses the given input.

        Args:
            tensor_in: A `Tensor` of shape=(batch_size, height,
                width, channel) representing an input image.
            output_shape: The shape of the output (H x W)

        Returns:
            Preprocessed tensor input, resized to the output_size
        """

        # Only reshape if input shape does not match
        if not tensor_in.shape[1:3] == output_shape:
            return tf.image.resize_images(tensor_in, output_shape)

        return tensor_in

    @abstractmethod
    def build(self, **kwargs):
        pass

