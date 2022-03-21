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
import tensorflow as tf

# a small value
EPSILON = 1e-8

# this is used when we are doing box encoding/decoding
SCALE_FACTORS = [10.0, 10.0, 5.0, 5.0]
# you can read about them here:
# github.com/rykov8/ssd_keras/issues/53
# github.com/weiliu89/caffe/issues/155

# here are input pipeline settings.
# you need to tweak these numbers for your system,
# it can accelerate training
SHUFFLE_BUFFER_SIZE = 15000
NUM_THREADS = 8
# read here about the buffer sizes:
# stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle

# images are resized before feeding them to the network
RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR

# threshold for IoU when creating training targets
MATCHING_THRESHOLD = 0.35

# this is used in tf.map_fn when creating training targets or doing NMS
PARALLEL_ITERATIONS = 8

# this can be important
BATCH_NORM_MOMENTUM = 0.9

