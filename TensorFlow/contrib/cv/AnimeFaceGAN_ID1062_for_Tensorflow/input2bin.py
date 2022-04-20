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
from utils import truncated_noise_sample, check_dir
import numpy as np
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument("--gen_num", type=int, default=5000, help="number of generated images")
    parser.add_argument("--output", type=str, default="../output", help="output path")
    parser.add_argument("-c", "--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument("--img_h", type=int, default=32, help="image height")
    parser.add_argument("--img_w", type=int, default=32, help="image width")
    parser.add_argument("--train_img_size", type=int, default=32,
                        help="image will be resized to this size when training")
    # model arguments
    parser.add_argument("--z_dim", type=int, default=120, help="latent space dimensionality")
    parser.add_argument("--truncation", type=float, default=2.0, help="truncation threshold")
    args = parser.parse_args()

    bin_path = os.path.join(args.output, "input_bin", str(args.train_img_size))
    z_bin_path = os.path.join(bin_path, "z")
    y_bin_path = os.path.join(bin_path, "y")
    check_dir(z_bin_path)
    check_dir(y_bin_path)

    for i in range(args.gen_num):
        z = truncated_noise_sample(1, args.z_dim, args.truncation)
        y = np.random.randint(args.num_classes, size=(1, 1))
        z.tofile(os.path.join(z_bin_path, str(i) + ".bin"))
        y.tofile(os.path.join(y_bin_path, str(i) + ".bin"))
