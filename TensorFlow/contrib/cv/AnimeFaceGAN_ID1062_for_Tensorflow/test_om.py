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
import argparse
import os
from PIL import Image
from utils import restore_img, check_dir, read_images
from calc_IS_FID import get_FID, get_IS
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="../output", help="output path")
    parser.add_argument("--train_img_size", type=int, default=32,
                        help="image will be resized to this size when training")
    parser.add_argument("--chip", type=str, default="gpu", help="run on which chip, cpu or gpu or npu")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use (leave blank for CPU only)")
    parser.add_argument("--batch_size", type=int, default=100, help="batch size")
    parser.add_argument("--precalculated_path", type=str, default="./metrics/res/stats_tf/fid_stats_cifar10_train.npz",
                        help="precalculated statistics for datasets, used in FID")
    args = parser.parse_args()

    bin_path = os.path.join(args.output, "inference", str(args.train_img_size), "bin")
    image_path = os.path.join(args.output, "inference", str(args.train_img_size), "image")
    check_dir(image_path)

    # recover image from bin
    print("Recovering image from bin...")
    files = os.listdir(bin_path)
    output_num = 0
    for file_name in tqdm(files):
        if file_name.endswith(".bin"):
            output_num += 1
            file_bin_path = os.path.join(bin_path, file_name)
            file_image_path = os.path.join(image_path, file_name.replace(".bin", ".jpg"))
            image = np.fromfile(file_bin_path, dtype='float32').reshape(args.train_img_size, args.train_img_size, 3)
            Image.fromarray(np.uint8(restore_img(image))).save(file_image_path)

    # calc FID and IS
    print("Calculating FID and IS...")
    images_list = read_images(image_path)
    images = np.array(images_list).astype(np.float32)
    fid_score = get_FID(images, args)
    is_mean, is_std = get_IS(images_list, args, splits=10)
    print("IS : (%f, %f)" % (is_mean, is_std))
    print("FID : %f" % fid_score)
