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
import os
import acl_utils as utils
import sys
import cv2 as cv


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default=r'datasets/kodak',
                        type=str,
                        help='input file path.')
    parser.add_argument('-o', '--output', default=r'bin/kodak', type=str,
                        help='Output folder.')
    parser.add_argument('--noise', default='gaussian', type=str, help='Type of noise corruption (one of: gaussian, poisson)')
    parser.add_argument('--width', type=int, default=768, help='resized image width before inference.')
    parser.add_argument('--height', type=int, default=512, help='resized image height before inference.')

    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


def resize_img(args, input_img):
    old_size = input_img.shape[0:2]
    target_size = [args.height, args.width]
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i * ratio) for i in old_size])
    img_new = cv.resize(input_img, (new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    resized_img = cv.copyMakeBorder(img_new, top, bottom, left, right, cv.BORDER_CONSTANT, None, (0, 0, 0))
    # resized_img = cv.resize(input_img, (args.width, args.height))
    return resized_img


def eprint(*args, **kwargs):
    """print error message to stderr
    """
    print(*args, file=sys.stderr, **kwargs)


def check_args(args):
    """check console parameters according to restrictions.
    :return: True or False
    """
    check_flag = True
    is_dir = True
    if os.path.isdir(args.input):
        if not os.listdir(args.input):
            eprint('[ERROR] input image path=%r is empty.' % args.input)
            check_flag = False
    elif os.path.isfile(args.input):
        is_dir = False
    else:
        eprint('[ERROR] input path=%r does not exist.' % args.input)
        check_flag = False
    return check_flag, is_dir


def mkdir_output(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    return


def noise(args, x):
    if args.noise is not None:
        if args.noise == "gaussian":
            x = utils.add_validation_gaussian_noise_np(x, validation_stddev=25)
        if args.noise == "poisson":
            x = utils.add_validation_poisson_noise_np(x)
    return x


def img2bin(args, file_path):
    try:
        # print("1")
        im = cv.imread(file_path)

        # if im.shape[0] != 512 and im.shape[1] != 768:
        #     im = im.transpose([1, 0, 2])
        im = resize_img(args, im)

        # cv.imshow('img_window', im)  # 显示图片,[图片窗口名字，图片]
        # cv.waitKey(0)  # 无限期显示窗口

        reshaped = im.transpose([2, 0, 1]) / 255.0 - 0.5
        reshaped = noise(args, reshaped)
        np.expand_dims(reshaped, axis=0)
        reshaped = reshaped.astype(np.float32)

        # reshaped = np.clip((reshaped + 0.5) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        # cv.imshow('img_window', im)  # 显示图片,[图片窗口名字，图片]
        # cv.waitKey(0)  # 无限期显示窗口

        out_path = os.path.join(args.output, os.path.splitext(os.path.split(file_path)[1])[0] + ".bin")
        mkdir_output(args)
        reshaped.tofile(out_path)

    except OSError as e:
        print('[ERROR] Skipping file', file_path, 'due to error: ', e)


def main():
    print('[info] Start changing npy images to bin...')
    args = parse_args()
    ret, is_dir = check_args(args)
    if ret:
        if is_dir:
            files_name = os.listdir(args.input)
            for file_name in files_name:
                file_path = os.path.join(args.input, file_name)
                img2bin(args, file_path)
        else:
            img2bin(args, args.input)
        print("[info] bin file generated successfully.")


if __name__ == '__main__':
    main()
