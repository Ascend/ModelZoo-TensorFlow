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
import  scipy
import argparse
import os
import sys


def parse_args():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        default=r'result/ixi_valid', type=str,
                        help='input file path.')
    parser.add_argument('--output', default=r'img/ixi_valid', type=str, help='Output file.')
    parser.add_argument('--width', type=int, default=255, help='resized image width before inference.')
    parser.add_argument('--height', type=int, default=255, help='resized image height before inference.')
    parser.add_argument('--bs', default=16, type=int, help='Batch size.')
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


def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x


def bin2img(args, file_path):
    image = np.fromfile(file_path, dtype=np.float32)
    image = np.reshape(image, (args.bs, args.height, args.width))
    mkdir_output(args)
    for i in range(image.shape[0]):
        prim = [image[i]]
        pimg = np.concatenate(prim, axis=1) + 0.5
        img = pimg

        # 傅里叶变换 可注释不进行展示
        # spec = [fftshift2d(abs(np.fft.fft2(x))) for x in prim]
        # simg = np.concatenate(spec, axis=1) * 0.03
        # img = np.concatenate([pimg, simg], axis=0)

        # cv.imshow('img_window', image)  # 显示图片,[图片窗口名字，图片]
        # cv.waitKey(0)  # 无限期显示窗口

        out_path = os.path.join(args.output, os.path.splitext(os.path.split(file_path)[1])[0] + '_' + str(i) + ".png")

        scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(out_path)


if __name__ == '__main__':
    main()
