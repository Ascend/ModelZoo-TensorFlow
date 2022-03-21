"""LICENSE"""
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
import imageio
import numpy as np
import argparse
from PIL import Image
import os

from numba import uint8

"""  transform .bin file to .jpg picture.

python3 bin2jpg.py --data_dir " ./datasets/" --dst_dir "./output/" 

"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./result_bin/')
    parser.add_argument("--dst_dir", type=str, default='./results_image/2736/')
    config = parser.parse_args()
#
    if os.path.exists(config.dst_dir):
        pass
    else:
        os.makedirs(config.dst_dir)
    num = 1
    for file in os.listdir(config.data_dir):
        if file.endswith('.bin'):
            data_dir = config.data_dir + "/" + file
            data = np.fromfile(data_dir, dtype='float32')
            data = data.reshape(268, 360)
            img = Image.fromarray((data * 255).astype('float32')).convert('L')
            img.save(config.dst_dir + "/" + str(num) + ".bmp")
            num = num + 1