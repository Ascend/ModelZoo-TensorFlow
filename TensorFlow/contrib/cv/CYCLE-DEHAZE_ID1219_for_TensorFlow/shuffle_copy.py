# MIT License

# Copyright (c) 2018 Deniz Engin

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
"""
The purpose of this file is to divide the complete data set into training set and test set, and copy to the corresponding directory
"""
from npu_bridge.npu_init import *
import random
import shutil
import os


def main():
    '''
    divide the complete data set into training set and test set
    '''
    img_num = 1449
    img_num_list = list(range(img_num))
    random.shuffle(img_num_list)
    test_num = 200
    all_directory = "data/allData"
    train_directory = "data/trainData"
    test_directory = "data/testData"
    os.mkdir(train_directory)
    os.mkdir(test_directory)
    os.mkdir(train_directory + "/hazyImage")
    os.mkdir(train_directory + "/clearImage")
    os.mkdir(test_directory + "/hazyImage")
    os.mkdir(test_directory + "/groundtruth")
    for i in range(test_num):
        num_str = str(img_num_list[i]).zfill(4)
        num_str += ".png"
        shutil.copy(all_directory + "/hazyImage/" + num_str, test_directory + "/hazyImage/" + num_str)
        shutil.copy(all_directory + "/clearImage/" + num_str, test_directory + "/groundtruth/" + num_str)
    for i in range(test_num, img_num):
        num_str = str(img_num_list[i]).zfill(4)
        num_str += ".png"
        shutil.copy(all_directory + "/hazyImage/" + num_str, train_directory + "/hazyImage/" + num_str)
        shutil.copy(all_directory + "/clearImage/" + num_str, train_directory + "/clearImage/" + num_str)


if __name__ == '__main__':
    main()
