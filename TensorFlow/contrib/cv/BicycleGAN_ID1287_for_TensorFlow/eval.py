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
from tqdm import tqdm
import tensorflow as tf
import lpips_tf
from PIL import Image
from npu_bridge.npu_init import *
from tensorflow_core.core.protobuf.rewriter_config_pb2 import RewriterConfig


def eval_tf(basedir):
    # NPU config
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

    with tf.Session(config=config) as sess:
        image_dirs = []
        for root, dirs, files in os.walk(basedir):  # find all dirs
            if dirs != []:
                for dirname in dirs:
                    full_dirname = os.path.join(root, dirname)
                    image_dirs.append(full_dirname)

        dist_consecutive = []
        image0_ph = tf.placeholder(tf.float32)
        image1_ph = tf.placeholder(tf.float32)
        dist_t = lpips_tf.lpips(image0_ph, image1_ph)

        for dir in tqdm(image_dirs):  # find all pictures of the dir
            lpips_pairs = []
            files = os.listdir(dir)
            for file in files:
                if file.startswith('random'):
                    path = os.path.join(dir, file)
                    image = Image.open(path)
                    image = np.asarray(image.resize((256, 256), Image.BICUBIC))
                    # when evaluating,the image is normalized to [0,1],
                    # because the lpips will do the work that transforms [0,1] to [-1,1]
                    image = image.astype(np.float32) / 255.0
                    lpips_pairs.append(image)

            for i in range(0, len(lpips_pairs) - 1):  # consecutive test,computing (N-1) pairs
                dist = sess.run(dist_t, feed_dict={image0_ph: lpips_pairs[i], image1_ph: lpips_pairs[i + 1]})
                dist_consecutive.append(dist)

        print('Final Average Distances : {}'.format(sum(dist_consecutive) / len(dist_consecutive)))
