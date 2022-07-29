#coding=utf-8
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
'''
Created on 2016骞10鏈12鏃

@author: dengdan
'''
from npu_bridge.npu_init import *
import datetime
import logging

import sys

def get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d %H:%M:%S')  

def init_logger(log_file = None, log_path = None, log_level = logging.DEBUG, mode = 'w', stdout = True):
    """
    log_path: 鏃ュ織鏂囦欢鐨勬枃浠跺す璺緞
    mode: 'a', append; 'w', 瑕嗙洊鍘熸枃浠跺啓鍏.
    """
    import util
    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_path is None:
        log_path = '~/temp/log/' 
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.log'
    log_file = util.io.join_path(log_path, log_file)
    # 姝ゅ涓嶈兘浣跨敤logging杈撳嚭
    print('log file path:' + log_file);
    util.io.make_parent_dir(log_file)
    logging.basicConfig(level = log_level,
                format= fmt,
                filename= util.io.get_absolute_path(log_file),
                filemode=mode)
    
    if stdout:
        console = logging.StreamHandler(stream = sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

#     console = logging.StreamHandler(stream = sys.stderr)
#     console.setLevel(log_level)
#     formatter = logging.Formatter(fmt)
#     console.setFormatter(formatter)
#     logging.getLogger('').addHandler(console)


