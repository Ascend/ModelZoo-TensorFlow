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
# Author: Salli Moustafa (salli.moustafa@huawei.com)
"""
ATS Framework

Task scheduler
"""

from concurrent.futures import ThreadPoolExecutor, wait
import threading

from ats.constants import MAX_DEVICE_COUNT


class TaskScheduler(object):
    """
    A generic task scheduler that allows to distribute inference tasks over several devices
    """

    def __init__(self):
        self.futures = [[] for _ in range(MAX_DEVICE_COUNT)]
        self.device_count = -1
        self.executors = None
        self.executors_ids = None

    def init(self, device_count):
        """
        Initializes the scheduler:
            - start the thread pools
        """
        self.device_count = device_count

        self.executors = [ThreadPoolExecutor(max_workers=1) for _ in range(device_count)]

        self.executors_ids = [None]*device_count
        for device_id in range(self.device_count):
            executor_id = self.executors[device_id].submit(threading.get_ident).result()
            if executor_id is not None:
                self.executors_ids[device_id] = executor_id
            else:
                raise Exception("Failed to get executor ID on device: {}".format(device_id))

    def finalize(self):
        """
        Finalizes the scheduler:
            - stop the thread pools
        """
        for device_id in range(self.device_count):
            self.executors[device_id].shutdown()

    def schedule(self, callable_task, device_id, *args, **kwargs):
        """
        Schedule the callable for being processed on the specified device
        """
        self.futures[device_id].append(self.executors[device_id].submit(callable_task,
                                                                        device_id,
                                                                        *args,
                                                                        **kwargs))

    def synchronize(self, device_id):
        """
        Wait until all tasks on the specified have been completed
        """
        wait(self.futures[device_id])

    def get_id(self, device_id):
        """
        Return the id of the thread managing the device with an id of device_id
        """
        return self.executors_ids[device_id]
