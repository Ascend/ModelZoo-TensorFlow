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
import os

import cv2
import numpy as np


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True, type=str, \
                   help='folder of input image or file of other input.')
    p.add_argument('-o', '--output', default='./output', \
                   help='output path.')
    return p.parse_args()


def mkdir_output(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    return


def bin2img(args, bin_file_path):
    infer = np.fromfile(bin_file_path, dtype=np.float32)
    infer = np.reshape(infer, (96, 96, 3))
    infer *= 255
    mkdir_output(args)
    cv2.imwrite(os.path.join(args.output, (os.path.splitext(os.path.split(bin_file_path)[1])[0] + '.jpg')), infer)


def main():
    args = get_args()
    bin_files = os.listdir(args.input)
    for bin_file in bin_files:
        bin_file_path = os.path.join(args.input, bin_file)
        bin2img(args, bin_file_path)
    print("[info] Transfer bin file into jpg successfully.")


if __name__ == '__main__':
    main()
