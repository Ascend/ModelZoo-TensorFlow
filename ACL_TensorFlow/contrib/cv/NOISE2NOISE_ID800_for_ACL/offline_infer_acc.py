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


import os
import argparse
import cv2 as cv
import numpy as np
from acl_utils import clip_to_uint8, chw_to_hwc


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--bs', default=1,type=int,
                        help='batchsize')
    parser.add_argument('--bin_dir', default='result/kodak', type=str,
                        help="""the bin data path""")
    parser.add_argument('--dataset', default='dataset/kodak', type=str,
                        help="""the nyu data path""")
    parser.add_argument('--width', type=int, default=768, help='resized image width before inference.')
    parser.add_argument('--height', type=int, default=512, help='resized image height before inference.')
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) > 0:
        for bad_arg in unknown_args:
            print("ERROR: Unknown command line arg: %s" % bad_arg)
        raise ValueError("Invalid command line arg(s)")
    return args


def read_bin_file(file_path):
    bin_files = os.listdir(file_path)
    loaded_dic = {}
    for bin_file in bin_files:
        file_name = bin_file.split("_")[0]
        loaded_dic.update({file_name: bin_file})

    return loaded_dic


def read_dataset_file(file_path):
    dataset_files = os.listdir(file_path)
    loaded_dic = {}
    for dataset_file in dataset_files:
        file_name = dataset_file.split(".")[0]
        loaded_dic.update({file_name: dataset_file})

    return loaded_dic


def load_image(image_path):
    im = cv.imread(image_path)
    return im


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


def main():
    args = parse_args()
    # data load and preprocess
    # load bin data
    print("bin images data loaded")

    # load dataset test data
    dataset_path = read_dataset_file(args.dataset)
    files_name = os.listdir(args.bin_dir)

    avg_psnr = 0.0
    w = args.width
    h = args.height

    for file_name in files_name:
        file_path = os.path.join(args.bin_dir, file_name)

        pred = np.fromfile(file_path, dtype=np.float32)
        pred = np.reshape(pred, (1, 3, args.height, args.width))
        pred = chw_to_hwc(clip_to_uint8(pred[0]))

        orig_path = os.path.join(args.dataset, dataset_path[file_name.split("_")[0]])
        orig = load_image(orig_path)
        orig = resize_img(args, orig)

        sqerr = np.square(orig.astype(np.float32) - pred.astype(np.float32))
        s = np.sum(sqerr)
        cur_psnr = 10.0 * np.log10((255 * 255) / (s / (w * h * 3)))
        avg_psnr += cur_psnr

    avg_psnr /= len(files_name)
    print('Average PSNR: %.2f' % avg_psnr)


if __name__ == '__main__':
    main()
