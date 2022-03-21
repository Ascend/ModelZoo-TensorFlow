# ============================================================================
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
from acl_utils import clip_to_uint8, chw_to_hwc
import argparse
import cv2 as cv
import os
import sys


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        default=r'result/kodak', type=str,
                        help='input file path.  folder path/ image file path')
    parser.add_argument('--output', default=r'img/kodak', type=str, help='Output file.')
    parser.add_argument('--width', type=int, default=768, help='resized image width before inference.')
    parser.add_argument('--height', type=int, default=512, help='resized image height before inference.')
    parser.add_argument('--bs', default=1, type=int, help='Batch size.')
    args = parser.parse_args()
    return args


def check_args(args):
    """check console parameters according to restrictions.
    :return: True or False
    """
    check_flag = True
    is_dir = True
    if os.path.isdir(args.input):
        if not os.listdir(args.input):
            eprint('[ERROR] input bin path=%r is empty.' % args.input)
            check_flag = False
    elif os.path.isfile(args.input):
        is_dir = False
    else:
        eprint('[ERROR] input path=%r does not exist.' % args.input)
        check_flag = False
    return check_flag, is_dir


def eprint(*args, **kwargs):
    """print error message to stderr
    """
    print(*args, file=sys.stderr, **kwargs)


def main():
    print('[info] Start changing bin images to image...')
    args = parse_args()
    ret, is_dir = check_args(args)
    if ret:
        if is_dir:
            files_name = os.listdir(args.input)
            for file_name in files_name:
                file_path = os.path.join(args.input, file_name)
                bin2img(args, file_path)
        else:
            bin2img(args, args.input)
        print("[info] image file generated successfully.")


def mkdir_output(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    return


def bin2img(args, file_path):
    image = np.fromfile(file_path, dtype=np.float32)
    image = np.reshape(image, (1, 3, args.height, args.width))
    image = chw_to_hwc(clip_to_uint8(image[0]))

    # cv.imshow('img_window', image)  # 显示图片,[图片窗口名字，图片]
    # cv.waitKey(0)  # 无限期显示窗口

    out_path = os.path.join(args.output, os.path.splitext(os.path.split(file_path)[1])[0] + ".png")
    mkdir_output(args)

    cv.imwrite(out_path, image)


if __name__ == '__main__':
    main()
