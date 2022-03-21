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
import sys

import numpy as np

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_bin_dir', type=str,
                        help='dir where to output bin.',default= "./Bin")
    return parser.parse_args(argv)

def main(args):
    fake_image = np.random.uniform(size=(256, 100))  #Generate random noises as fake images
    fake_image = fake_image.astype("float32")   # specify precision
    print(fake_image,fake_image.dtype)
    fake_image.tofile(args.output_bin_dir + "/fake_image.bin")
    label = np.random.randint(10, size=256)  #Generate labels for numbers
    label = label.astype("int32") # specify precision
    print(label,label.dtype)
    label.tofile(args.output_bin_dir + "/label.bin")
    
if __name__ == "__main__" :
    main(parse_arguments(sys.argv[1:]))