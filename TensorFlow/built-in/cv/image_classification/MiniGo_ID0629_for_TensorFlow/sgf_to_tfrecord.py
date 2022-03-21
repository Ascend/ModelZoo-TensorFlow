# -*- coding: UTF-8 -*-

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
# Copyright 2018 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import preprocessing
from absl import flags

# From dual_net.py
flags.declare_key_flag('input_features')
FLAGS = flags.FLAGS

def SGF_to_TFRecord(file_path):
    Filelist = []
    # traverse folders, return: current path, current path name, current path files name
    for presentdir, dirnames, filenames in os.walk(file_path):
        for filename in filenames:
            # files with path
            file_with_path = os.path.join(presentdir, filename)
            Filelist.append(file_with_path)

    for f in Filelist:
        try:
            preprocessing.make_dataset_from_sgf(f, f.replace(".sgf", ".tfrecord.zz"))
            print("sgf_to_tfrecord Success: " + f)
        except:
            print("sgf_to_tfrecord Fail: " + f)
            # remove fail file
            os.unlink(f)

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Convert sgf file to tfrecord file.")
    parser.add_argument('--sgf_path', type=str, help='sgf path')
    args = parser.parse_args()
    #print(args)

    # first of all, check arguments
    if args.sgf_path is None:
        print("\nPlease enter a sgf path before running the program !")

    SGF_to_TFRecord(args.sgf_path)
    print("sgf file to tfrecord file finished !")

if __name__ == '__main__':
    main()