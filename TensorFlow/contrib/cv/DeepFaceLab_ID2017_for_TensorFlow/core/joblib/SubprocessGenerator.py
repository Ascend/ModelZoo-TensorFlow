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
import queue as Queue
import threading
import time


class SubprocessGenerator(object):
    
    @staticmethod
    def launch_thread(generator): 
        generator._start()
        
    @staticmethod
    def start_in_parallel( generator_list ):
        """
        Start list of generators in parallel
        """
        for generator in generator_list:
            thread = threading.Thread(target=SubprocessGenerator.launch_thread, args=(generator,) )
            thread.daemon = True
            thread.start()

        while not all ([generator._is_started() for generator in generator_list]):
            time.sleep(0.005)
    
    def __init__(self, generator_func, user_param=None, prefetch=2, start_now=True):
        super().__init__()
        self.prefetch = prefetch
        self.generator_func = generator_func
        self.user_param = user_param
        self.sc_queue = multiprocessing.Queue()
        self.cs_queue = multiprocessing.Queue()
        self.p = None
        if start_now:
            self._start()

    def _start(self):
        if self.p == None:
            user_param = self.user_param
            self.user_param = None
            p = multiprocessing.Process(target=self.process_func, args=(user_param,) )
            p.daemon = True
            p.start()
            self.p = p
            
    def _is_started(self):
        return self.p is not None
        
    def process_func(self, user_param):
        self.generator_func = self.generator_func(user_param)
        while True:
            while self.prefetch > -1:
                try:
                    gen_data = next (self.generator_func)
                except StopIteration:
                    self.cs_queue.put (None)
                    return
                self.cs_queue.put (gen_data)
                self.prefetch -= 1
            self.sc_queue.get()
            self.prefetch += 1

    def __iter__(self):
        return self

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['p']
        return self_dict

    def __next__(self):
        self._start()
        gen_data = self.cs_queue.get()
        if gen_data is None:
            self.p.terminate()
            self.p.join()
            raise StopIteration()
        self.sc_queue.put (1)
        return gen_data
