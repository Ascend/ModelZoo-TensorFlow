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
import argparse
import numpy as np
import tensorflow as tf
import os
# download https://github.com/lzhbrian/metrics to calculate IS and FID
from metrics.inception_score_official_tf import get_inception_score
from metrics.fid_official_tf import calculate_activation_statistics, calculate_frechet_distance
from utils import read_images, session_config


def get_FID(images, arg):
    # load from precalculated
    f = np.load(args.precalculated_path)
    mu1, sigma1 = f['mu'][:], f['sigma'][:]
    f.close()

    # session configuration
    config = session_config(arg)

    # calc from image ndarray
    # images should be Numpy array of dimension (N, H, W, C). images should be in 0~255
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        mu2, sigma2 = calculate_activation_statistics(images, sess, batch_size=arg.batch_size)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def get_IS(images_list, arg, splits=10):
    return get_inception_score(images_list, splits=splits, sess_config=session_config(arg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chip", type=str, default="gpu", help="run on which chip, cpu or gpu or npu")
    parser.add_argument("--fake_img_path", type=str, default="../output/test/fake/32", help="fake image path")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use (leave blank for CPU only)")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    parser.add_argument("--precalculated_path", type=str, default="./metrics/res/stats_tf/fid_stats_cifar10_train.npz",
                        help="precalculated statistics for datasets, used in FID")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    image_list = read_images(args.fake_img_path)
    image = np.array(image_list).astype(np.float32)

    fid_score = get_FID(image, args)
    is_mean, is_std = get_IS(image_list, args, splits=10)

    print("IS : (%f, %f)" % (is_mean, is_std))
    print("FID : %f" % fid_score)
