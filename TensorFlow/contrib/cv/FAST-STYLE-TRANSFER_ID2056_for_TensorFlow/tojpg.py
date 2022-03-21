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
import imageio

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", default="./datasets/imagenet_10", help="path of input bin pictures")
    parser.add_argument("--dst_path", default="./input", help="path of output jpg files")

    args = parser.parse_args()
    src_path = args.src_path
    dst_path = args.dst_path

    data = np.fromfile(src_path, np.float32).reshape(256, 256, 3)
    data = np.clip(data, 0, 255).astype(np.uint8)
    imageio.imsave(dst_path+"/" + "1" + ".jpg", data)