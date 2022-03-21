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

Generic interface between the user and the scheduler
"""

from ats.acl_resource import ACLResourceManager
from ats.scheduler import TaskScheduler


class Engine(object):
    """
    Generic interface between the user and the scheduler
    """

    def __init__(self, device_count, stream_count, profiling_mode=False):
        self.device_count = device_count
        self.stream_count = stream_count

        self.acl_resources = [None for _ in range(device_count)]
        for device_id in range(device_count):
            self.acl_resources[device_id] = ACLResourceManager(device_id,
                                                               device_count,
                                                               stream_count,
                                                               profiling_mode)

        self.scheduler = TaskScheduler()

    def launch(self, task, device_id=-1):
        """
        Submit the task to the scheduler. Each task must implement the AbstractTask interface
        """
        assert device_id < self.device_count

        if -1 == device_id:
            for _device_id in range(self.device_count):
                self.scheduler.schedule(task.init,
                                        _device_id,
                                        self.acl_resources[_device_id])

            for _device_id in range(self.device_count):
                self.scheduler.schedule(task.run,
                                        _device_id,
                                        self.acl_resources[_device_id])

            for _device_id in range(self.device_count):
                self.scheduler.schedule(task.finalize, _device_id, self.stream_count)
        else:
            self.scheduler.schedule(task.init,
                                    device_id,
                                    self.acl_resources[device_id])
            self.scheduler.schedule(task.run,
                                    device_id,
                                    self.acl_resources[device_id])
            self.scheduler.schedule(task.finalize, device_id, self.stream_count)

    def wait(self, device_id=-1):
        """
        Wait until the scheduler on the specified (or all) device(s) is ready to accept new tasks
        """
        if -1 == device_id:
            for _device_id in range(self.device_count):
                self.scheduler.synchronize(_device_id)
        else:
            self.scheduler.synchronize(device_id)

    def init(self, output_path):
        """
        Initializes the schedulers on all devices
        """
        self.scheduler.init(self.device_count)
        for device_id in range(self.device_count):
            self.scheduler.schedule(self.__init_device, device_id, output_path)

    def finalize(self):
        """
        Finalizes the schedulers on all devices
        """
        for device_id in range(self.device_count):
            self.scheduler.schedule(self.__finalize_device, device_id)
        self.scheduler.finalize()

    def __init_device(self, device_id, output_path):
        self.acl_resources[device_id].init(output_path)

    def __finalize_device(self, device_id):
        self.acl_resources[device_id].finalize()
