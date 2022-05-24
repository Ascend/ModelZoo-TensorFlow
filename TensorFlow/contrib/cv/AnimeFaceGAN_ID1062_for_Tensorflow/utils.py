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
import numpy as np
from scipy.stats import truncnorm
import tensorflow as tf
import imageio
from PIL import Image
import os
from glob import glob


def truncated_noise_sample(batch_size=1, dim_z=128, trunc=1., seed=None):
    """truncation trick"""
    state = None if seed is None else np.random.RandomState(seed)
    if trunc <= 0:
        return np.random.normal(size=(batch_size, dim_z))    # do not use truncation
    else:
        return truncnorm.rvs(-trunc, trunc, size=(batch_size, dim_z), random_state=state).astype(np.float32)


def read_image(filename):
    x = imageio.imread(filename)
    return np.array(Image.fromarray(x))


def read_images(img_path):
    filenames = glob(os.path.join(img_path, '*.*'))
    images_list = [read_image(filename) for filename in filenames]
    return images_list


def normalize_img(img):
    return img / 127.5 - 1


def restore_img(img):
    return (img + 1) * 127.5


def get_one_batch(data, labels, batch_size):
    rand_select = np.random.randint(0, data.shape[0], batch_size)
    batch_labels = labels[rand_select]
    batch = data[rand_select]

    return normalize_img(batch), batch_labels


def session_config(args):
    if args.chip == "npu":
        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        if args.use_fp16 is True:
            custom_op.parameter_map['precision_mode'].s = tf.compat.as_bytes('allow_mix_precision')
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        fusion_cfg_path = os.path.join(os.path.dirname(__file__), "fusion_switch.cfg")
        custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes(fusion_cfg_path)
        # custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")
        if args.profiling is True:
            custom_op.parameter_map["use_off_line"].b = True
            custom_op.parameter_map["profiling_mode"].b = True
            custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes(
                '{"output":"/tmp/profiling","task_trace":"on","aicpu":"on"}')
    elif args.chip == "gpu":
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
    elif args.chip == 'cpu':
        config = tf.ConfigProto()
    return config


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import scipy

    truncation = 3.0
    N = scipy.stats.norm(loc=0., scale=1.)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.hist(truncated_noise_sample(dim_z=10000, trunc=truncation).squeeze(), normed=True, bins=30)  # histogram of truncated normal distribution
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.hist(N.rvs(10000), normed=True, bins=30)  # histogram of standard normal distribution
    plt.show()
