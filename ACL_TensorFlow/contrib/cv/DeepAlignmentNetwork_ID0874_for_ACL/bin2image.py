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

import argparse
import cv2 as cv
import os
import sys


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_bin',
                        default=r'test/test.bin', type=str,
                        help='input file path.  folder path/ image file path')
    parser.add_argument('--input_img',
                        default=r'test/test.png', type=str,
                        help='input file path.  folder path/ image file path')
    parser.add_argument('--output', default=r'test/test_out.png', type=str, help='Output file.')
    parser.add_argument('--width', type=int, default=112, help='resized image width before inference.')
    parser.add_argument('--height', type=int, default=112, help='resized image height before inference.')
    parser.add_argument('--bs', default=1, type=int, help='Batch size.')
    args = parser.parse_args()
    return args


def check_args(args):
    """check console parameters according to restrictions.
    :return: True or False
    """
    check_flag = True
    is_dir = True
    if os.path.isdir(args.input_bin):
        if not os.listdir(args.input_bin):
            eprint('[ERROR] input bin path=%r is empty.' % args.input_bin)
            check_flag = False
    elif os.path.isfile(args.input_bin):
        is_dir = False
    else:
        eprint('[ERROR] input bin path=%r does not exist.' % args.input_bin)
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
            files_name = os.listdir(args.input_bin)
            for file_name in files_name:
                file_path = os.path.join(args.input_bin, file_name)
                bin2img(args, file_path)
        else:
            bin2img(args, args.input_bin)
        print("[info] image file generated successfully.")


def mkdir_output(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    return


def bin2img(args, bin_file_path):
    infer = np.fromfile(bin_file_path, dtype=np.float32)
    infer = np.reshape(infer, (68, 2))
    infer = infer.astype(np.uint8)

    try:
        img = cv.imread(os.path.join(args.input_img,
                                     os.path.split(bin_file_path)[1].split('_output_')[0] + ".png")) if os.path.isdir(
            args.input_img) else cv.imread(
            args.input_img)
    except Exception as e:
        print(e)
        img = np.zeros((args.height, args.width))

    for point in infer:
        cv.circle(img, point, 0, (0, 255, 0), -1)

    # cv.namedWindow("image")
    # cv.imshow('image', img)
    # cv.waitKey(10000)  # 显示 10000 ms 即 10s 后消失
    # cv.destroyAllWindows()

    # cv.imshow('img_window', image)  # 显示图片,[图片窗口名字，图片]
    # cv.waitKey(0)  # 无限期显示窗口

    out_path = os.path.join(args.output, os.path.splitext(os.path.split(bin_file_path)[1])[0] + ".png")
    mkdir_output(args)

    cv.imwrite(out_path, img)


if __name__ == '__main__':
    main()
