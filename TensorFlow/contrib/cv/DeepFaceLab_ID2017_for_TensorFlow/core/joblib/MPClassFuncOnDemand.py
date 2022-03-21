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

import multiprocessing
from core.interact import interact as io


class MPClassFuncOnDemand():
    def __init__(self, class_handle, class_func_name, **class_kwargs):
        self.class_handle = class_handle
        self.class_func_name = class_func_name
        self.class_kwargs = class_kwargs
   
        self.class_func = None
        
        self.s2c = multiprocessing.Queue()
        self.c2s = multiprocessing.Queue()
        self.lock = multiprocessing.Lock()
        
        io.add_process_messages_callback(self.io_callback)

    def io_callback(self):        
        while not self.c2s.empty():
            func_args, func_kwargs = self.c2s.get()
            if self.class_func is None:
                self.class_func = getattr( self.class_handle(**self.class_kwargs), self.class_func_name)
            self.s2c.put ( self.class_func (*func_args, **func_kwargs) )

    def __call__(self, *args, **kwargs):
        with self.lock:
            self.c2s.put ( (args, kwargs) )
            return self.s2c.get()

    def __getstate__(self):
        return {'s2c':self.s2c, 'c2s':self.c2s, 'lock':self.lock}

