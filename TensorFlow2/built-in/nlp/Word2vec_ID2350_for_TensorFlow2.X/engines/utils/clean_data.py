# -*- coding: utf-8 -*- 
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
# @Time : 2021/3/17 16:37  
# @Author : Stanley  
# @EMail : gzlishouxian@gmail.com
# @File : clean_data.py
# @Software: PyCharm
# import npu_device
# npu_device.open().as_default()
import re


def filter_word(raw_word):
    if not re.search(r'^[\u4e00-\u9fa5_a-zA-Z0-9]+$', raw_word):
        return False
    else:
        return True


def filter_char(char):
    if not re.search(r'[\u4e00-\u9fa5_a-zA-Z0-9]', char):
        return False
    else:
        return True

