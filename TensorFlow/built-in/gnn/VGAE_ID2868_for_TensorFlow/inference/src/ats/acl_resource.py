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
# Author: Salli Moustafa 
"""
ATS Framework

ACL resource management
"""

import acl
from ats.utils import check_status
from ats.constants import ACL_PROF_ACL_API
from ats.constants import ACL_PROF_AICORE_METRICS
from ats.constants import ACL_AICORE_PIPE_UTILIZATION
from ats.constants import ACL_PROF_TASK_TIME


class ACLResourceManager(object):
    """
    Manages ACL resources (devices, context, streams)
    """

    def __init__(self, device_id=0, device_count=1, stream_count=1, profiling_mode=False):
        self.device_id = device_id
        self.device_count = device_count
        self.stream_count = stream_count
        self.context = None
        self.streams = [None for _ in range(stream_count)]
        self.run_mode = None
        self.profiling_mode = profiling_mode
        self.profiling_config = None

    def init(self, output_path):
        """
        Initializes ACL resources
        """
        print("[device {}] init resource stage:".format(self.device_id))

        if self.profiling_mode:
            status = acl.prof.init(output_path + "/profiling")
            self.profiling_config = \
                acl.prof.create_config([0],
                                       ACL_AICORE_PIPE_UTILIZATION,
                                       0,
                                       ACL_PROF_ACL_API |
                                       ACL_PROF_AICORE_METRICS |
                                       ACL_PROF_TASK_TIME)
            status = acl.prof.start(self.profiling_config)

        status = acl.rt.set_device(self.device_id)
        check_status("acl.rt.set_device", status)

        self.context, status = acl.rt.create_context(self.device_id)
        check_status("acl.rt.create_context", status)

        for stream_id in range(self.stream_count):
            self.streams[stream_id], status = acl.rt.create_stream()
            check_status("acl.rt.create_stream", status)

        self.run_mode, status = acl.rt.get_run_mode()
        check_status("acl.rt.get_run_mode", status)

        print("[device {}] Init resource success".format(self.device_id))

    def finalize(self):
        """
        Finalizes ACL resources
        """
        for stream in self.streams:
            if stream:
                status = acl.rt.destroy_stream(stream)
                check_status("acl.rt.destroy_stream", status)

        if self.context:
            status = acl.rt.destroy_context(self.context)
            check_status("acl.rt.destroy_context", status)

        acl.rt.reset_device(self.device_id)

        if self.profiling_mode:
            status = acl.prof.stop(self.profiling_config)
            status = acl.prof.destroy_config(self.profiling_config)
            status = acl.prof.finalize()

        print("[device {}] Release acl resource success".format(self.device_id))
