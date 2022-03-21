# -*- coding: utf-8 -*-

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
import os,re

def Get_FileList(file_path):
    Filelist = []
    # traverse folders, return: current path, current path name, current path files name
    for presentdir, dirnames, filenames in os.walk(file_path):
        for filename in filenames:
            # files with path
            file_with_path = os.path.join(presentdir, filename)
            Filelist.append(file_with_path)
    return Filelist

def Read_and_New_File(file):
    file_line = []
    try:
        # gb18030 encode and ignore errors
        for line in open(file, encoding='gb18030', errors='ignore'):
            # ignore chinese
            line = re.sub('[\u4E00-\u9FA5]',"", line)
            file_line.append(line)
        os.unlink(file)
        with open(file,'w') as F:
            F.writelines(file_line)
            F.close()
        print("New File Success: " + file)
    except:
        print("New File Fail: " + file)
        # remove fail file
        os.unlink(file)

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Check sgf file, convert chinese to ''.")
    parser.add_argument('--sgf_path', type=str, help='sgf path')
    args = parser.parse_args()
    #print(args)

    # first of all, check arguments
    if args.sgf_path is None:
        print("\nPlease enter a sgf path before running the program !")

    sgf_list = Get_FileList(args.sgf_path)
    for file in sgf_list:
        Read_and_New_File(file)
    print("Check sgf file finished !")

if __name__ == '__main__':
    main()