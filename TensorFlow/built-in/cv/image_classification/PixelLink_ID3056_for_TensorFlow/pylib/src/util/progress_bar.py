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
# code from http://python.jobbole.com/83692/
"""
from npu_bridge.npu_init import *
import sys, time

class ProgressBar:
    def __init__(self, total = 0, width = None, finish_symbol = '#',  init_symbol = '-', max_width = 100):
        self.count = 0
        self.total = total
        self.finish_symbol = finish_symbol
        self.init_symbol = init_symbol
        
        if width is not None:
            self.width = width
        else:
            self.width = total
            
        self.width = min(self.width, max_width)
            
        
    def move(self, count = 1, msg = ""):
        self.count += count
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        progress = self.width * self.count / self.total
        sys.stdout.write('{}{:3}/{:3}: '.format(msg, self.count, self.total))
        sys.stdout.write(self.finish_symbol * progress + self.init_symbol * (self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()



# bar = ProgressBar(100)
# for i in range(10):
#     bar.move(10)
#     time.sleep(0.1)
