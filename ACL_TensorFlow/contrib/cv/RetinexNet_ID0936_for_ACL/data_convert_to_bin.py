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
from glob import glob
import argparse
import numpy as np
from PIL import Image

def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test_dir', dest='test_dir', default='./test', help='directory for testing inputs')
    parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')

    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_dir = args.save_dir

    test_low_data_name = glob(os.path.join(args.test_dir) + '/*.*')
    test_low_data_names = glob(os.path.join(args.test_dir) + '/*.*')
    test_low_data = []

    for idx in range(len(test_low_data_name)):
        test_low_im = load_images(test_low_data_name[idx])
        test_low_data.append(test_low_im)

    for idx in range(len(test_low_data)):
        print(test_low_data_names[idx])
        [_, name] = os.path.split(test_low_data_names[idx])
        suffix = 'bin'
        name = name[:name.find('.')]

        input_low_test = np.expand_dims(test_low_data[idx], axis=0)
        print(os.path.join(save_dir, name + "_S." + suffix))
        size=input_low_test.shape
        input_low_test.tofile(os.path.join(save_dir, name + "_S_" + str(size[1]) + "_" + str(size[2]) + "." + suffix))
