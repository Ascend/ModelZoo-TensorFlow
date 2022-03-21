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
from pathlib import Path

'''
You can implement your own SampleGenerator
'''
class SampleGeneratorBase(object):


    def __init__ (self, debug=False, batch_size=1):
        self.debug = debug
        self.batch_size = 1 if self.debug else batch_size
        self.last_generation = None
        self.active = True

    def set_active(self, is_active):
        self.active = is_active

    def generate_next(self):
        if not self.active and self.last_generation is not None:
            return self.last_generation
        self.last_generation = next(self)
        return self.last_generation

    #overridable
    def __iter__(self):
        #implement your own iterator
        return self

    def __next__(self):
        #implement your own iterator
        return None
    
    #overridable
    def is_initialized(self):
        return True