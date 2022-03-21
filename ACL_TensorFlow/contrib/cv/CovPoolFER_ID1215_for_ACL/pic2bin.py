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
import sys

import numpy as np
from PIL import Image as im
import imageio
import os


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)
def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat
def get_dataset(paths):
    dataset = []
    for path in paths.split(':'):
        classes = os.listdir(path)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path, class_name)
            if os.path.isdir(facedir):
                images = os.listdir(facedir)
                images.sort()
                images = [img for img in images if img.endswith('.jpg') or img.endswith('.png')]
                image_paths = [os.path.join(facedir,img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))
    return dataset



def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_dir', type=str,
                        help='dir where to load images.',default= "/root/Neesky/SFEW_100/Val")
    parser.add_argument('--output_bin_dir', type=str,
                        help='dir where to output bin.',default= "/root/Neesky/SFEW_100/Bin")
    parser.add_argument('--output_reference_file', type=str,
                        help='file where to output reference(label) file.',default= "/root/Neesky/SFEW_100/Reference")
    return parser.parse_args(argv)
def main(args):
    train_set = get_dataset(args.input_image_dir)
    image_list, label_list = get_image_paths_and_labels(train_set)
    for i in range(len(label_list)):
        print(image_list[i], label_list[i])

    mydict = {}
    for i in range(len(image_list)):
        mydict[image_list[i].split('/')[-1]] = label_list[i]

    if (os.path.isdir(args.output_bin_dir) == False):
        os.mkdir(args.output_bin_dir)
    if (os.path.isdir(args.output_reference_file) == False):
        os.mkdir(args.output_reference_file)
    alltot = 0
    tot = 0
    out_put = np.zeros([128, 100, 100, 3], dtype=np.float32)
    reference_put = np.zeros([128], dtype=np.int)

    for base, _, names in os.walk(args.input_image_dir):
        for name in names:
            clear_image = im.open(os.path.join(base, name))
            # clear_image = clear_image.convert('RGB')
            clear_image_arr = np.array(clear_image).astype('float32')
            out_put[tot] = clear_image_arr
            reference_put[tot] = mydict[name]
            if tot == 127:
                out_put.tofile(args.output_bin_dir + '/{}.bin'.format(str(alltot)))
                with open(args.output_reference_file + '/{}.txt'.format(str(alltot)), "w") as file:
                    for re in reference_put:
                        file.write(str(re) + " ")
                alltot = alltot + 1
                tot = 0
            else:
                tot = tot + 1
            clear_image.close()

    print('ok')
if __name__ == "__main__" :
    main(parse_arguments(sys.argv[1:]))