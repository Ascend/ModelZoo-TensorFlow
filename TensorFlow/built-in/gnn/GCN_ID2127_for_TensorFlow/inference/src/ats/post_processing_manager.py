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

Post processing manager
"""

import acl

from ats.constants import MAX_DEVICE_COUNT
from ats.scheduler import TaskScheduler

from ats.constants import ACL_CALLBACK_NO_BLOCK


class CallbackData(object):
    # pylint: disable=too-few-public-methods
    """
    Holds the data to be passed to the callback
    """

    def __init__(self, run_mode, device_id, context, stream_id, stream, raw_data):
        self.run_mode = run_mode
        self.device_id = device_id
        self.context = context
        self.stream_id = stream_id
        self.stream = stream
        self.raw_data = raw_data


class PostProcessingManager(object):
    """
    Implement an asynchronous post-processing manager
    """

    def __init__(self, worker_size, timeout):
        self.worker_size = worker_size
        self.timeout = timeout

        if worker_size != 1:
            raise Exception(
                "PostProcessing currently supports a worker pool of one thread\n")

        self.processing_shutdown_requested = [False]*MAX_DEVICE_COUNT
        self.is_initialized = [False]*MAX_DEVICE_COUNT
        self.schedulers = [TaskScheduler()]*MAX_DEVICE_COUNT

    def init(self, device_id):
        """
        Initializes the post-processing manager on the specified device
            - start the per-device scheduler
        """
        if self.is_initialized[device_id]:
            return

        self.schedulers[device_id].init(self.worker_size)

        self.is_initialized[device_id] = True

    def finalize(self, device_id):
        """
        Finalizes the post-processing manager on the specified device
            - stop the per-device scheduler
        """
        if not self.is_initialized[device_id]:
            return

        self.schedulers[device_id].finalize()

        self.is_initialized[device_id] = False

    def subscribe(self, device_id, context, streams):
        """
        Subscribe the calling thread to the post-processing thread on the specified device
        """
        self.schedulers[device_id].schedule(self.__trigger, 0, context)

        thread_id = self.schedulers[device_id].get_id(0)
        for stream in streams:
            acl.rt.subscribe_report(thread_id, stream)

    def unsubscribe(self, device_id, streams):
        """
        Unsubscribe the calling thread from the post-processing thread on the specified device
        """
        thread_id = self.schedulers[device_id].get_id(0)
        for stream in streams:
            acl.rt.unsubscribe_report(thread_id, stream)

        self.processing_shutdown_requested[device_id] = True

    @staticmethod
    def launch(callback, run_mode, device_id, context, stream_id, stream, raw_data):
        """
        Asynchronously launch the post-processing callback
        """
        callback_data = [run_mode, device_id,
                         context, stream_id, stream, raw_data]

        return acl.rt.launch_callback(callback, callback_data, ACL_CALLBACK_NO_BLOCK, stream)

    def __trigger(self, device_id, context):
        acl.rt.set_context(context)

        while True:
            acl.rt.process_report(self.timeout)
            if self.processing_shutdown_requested[device_id]:
                return
