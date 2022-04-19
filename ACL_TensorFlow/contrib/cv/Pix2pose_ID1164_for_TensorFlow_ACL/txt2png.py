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
from PIL import Image
import argparse
import os
from glob import glob
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--atc_dir', dest='atc_dir', default='', help='directory for atc result')
    parser.add_argument('--width', dest='width', type=int, default=128)
    parser.add_argument('--height', dest='height', type=int, default=128)

    args = parser.parse_args()

    result = np.loadtxt(args.atc_dir, dtype=np.float)
    print(result.shape)
    result_1 = result.reshape(args.width, args.height, 3)
    print(result_1)
    # import pdb
    # pdb.set_trace()
    # im = Image.fromarray(np.clip(result_1 * 255.0, , 255.0).astype('uint8'))
    # im = Image.fromarray(np.uint8((result_1+1)/2 *255))
    # image=np.uint8((result_1 + 1) / 2 * 255)
    image = result_1
    plt.figure()
    plt.imshow(image)
    plt.show()



if __name__ == '__main__':
    main()