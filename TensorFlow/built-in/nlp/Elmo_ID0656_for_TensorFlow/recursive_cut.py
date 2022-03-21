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

# coding=UTF-8
#鍏ㄩ儴鍒囨垚涓夊瓧鍙婁互涓
from npu_bridge.npu_init import *
import jieba

def recursive_cut(line):
    result = []
    for big_word in jieba.lcut(line,HMM=False):
            subword_list = get_subword_list(big_word)
            if isinstance(subword_list, list):
                go_subword_list(subword_list,result)
            elif isinstance(subword_list, str):
                result.append(subword_list)
            else:
                print("error")
    return result

def isEN(uchar):
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False

def isZH(char):
    if not ('\u4e00' <= char <= '\u9fa5'):
        return False
    return True


def get_subword_list(big_word):
    if not isZH(big_word[0]):
        return big_word
    if len(big_word)>4:
        jieba.del_word(big_word)
        return jieba.lcut(big_word, HMM=False)
    else:
        return big_word

def go_subword_list(input_list,result):
    for big_word in input_list:
        if len(big_word)>4:
            subword_list = get_subword_list(big_word)
            if isinstance(subword_list,list):
                go_subword_list(subword_list,result)
            elif isinstance(subword_list,str):
                result.append(subword_list)
            else:
                print("error")
        else:
            result.append(big_word)

#print(recursive_cut("涓浜屼笁鍥涗簲鍏竷鍏節鍗"))
#print(recursive_cut("鍗佷節鍏竷鍏簲鍥涗笁浜屼竴"))
