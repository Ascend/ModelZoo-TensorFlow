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
from npu_bridge.npu_init import *
import time
import tqdm


class ShowProgress():
    """ This is a simple wrapper around tqdm that includes a starting delay before printing.
    """
    def __init__(self, iterable, total, desc, silent, start_delay):
        self.iter = iter(iterable)
        self.start_time = time.time()
        self.pbar = None
        self.total = total
        self.desc = desc
        self.start_delay = start_delay
        self.silent = silent
        self.unshown_count = 0
    
    def __next__(self):
        if self.pbar is None and time.time() - self.start_time > self.start_delay:
           self.pbar = tqdm.tqdm(total=self.total, initial=self.unshown_count, desc=self.desc, disable=self.silent)
           self.pbar.start_t = self.start_time
        if self.pbar is not None:
            self.pbar.update(1)
        else:
            self.unshown_count += 1
        try:
            return next(self.iter)
        except StopIteration as e:
            if self.pbar is not None:
                self.pbar.close()
            raise e

    def __iter__(self):
        return self

def show_progress(iterable, total=None, desc=None, silent=False, start_delay=10):
    return ShowProgress(iterable, total, desc, silent, start_delay)
