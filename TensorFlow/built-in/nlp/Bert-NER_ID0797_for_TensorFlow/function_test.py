#! usr/bin/env python3
# -*- coding:utf-8 -*-
#
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
#
"""
@Author:zhoukaiyin
"""


def _read_data(input_file):
    """Read a BIO data!"""
    rf = open(input_file, 'r')
    lines = []; words = []; labels = []
    for line in rf:
        word = line.strip().split(' ')[0]
        label = line.strip().split(' ')[-1]
        # here we dont do "DOCSTART" check
        if len(line.strip()) == 0 and words[-1] == '.':
            l = ' '.join([label for label in labels if len(label) > 0])
            w = ' '.join([word for word in words if len(word) > 0])
            lines.append((l, w))
            words = []
            labels = []
        words.append(word)
        labels.append(label)
    return lines

def main():
   lines =  _read_data("./data/train.txt")
   print(lines)
main()
