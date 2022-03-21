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
import re

PATTERN = ('.jpg', '.jpeg')


def find_files(directory, pattern=PATTERN):
    files = []
    for path, d, filelist in os.walk(directory):
        for filename in filelist:
            if filename.lower().endswith(pattern):
                files.append(os.path.join(path, filename))
    return files


def map2classnames(labelmap_file):
    classes = []
    f = open(labelmap_file, 'r')
    pat = 'display_name'
    for line in f.readlines():
        if re.search(pat, line):
            line_strs = line.split('"')
            class_name = line_strs[-2]
            classes.append(class_name)
    f.close()
    return classes
