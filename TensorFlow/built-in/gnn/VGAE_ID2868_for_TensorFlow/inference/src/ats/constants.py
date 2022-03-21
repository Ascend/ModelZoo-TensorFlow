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

Common constants
"""

MAX_DEVICE_COUNT = 32
ATS_SEND = 1
ATS_RECV = 2

ACL_DEVICE = 0
ACL_HOST = 1

# Callback management
ACL_CALLBACK_NO_BLOCK = 0
ACL_CALLBACK_BLOCK = 1

# profiling modes
ACL_PROF_ACL_API = 0x0001
ACL_PROF_TASK_TIME = 0x0002
ACL_PROF_AICORE_METRICS = 0x0004
ACL_PROF_AICPU = 0x0008

# AI Core metrics
ACL_AICORE_ARITHMETIC_UTILIZATION = 0
ACL_AICORE_PIPE_UTILIZATION = 1
ACL_AICORE_MEMORY_BANDWIDTH = 2
ACL_AICORE_L0B_AND_WIDTH = 3
ACL_AICORE_RESOURCE_CONFLICT_RATIO = 4
ACL_AICORE_NONE = 0xFF

# Memory allocation policies
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEM_MALLOC_HUGE_ONLY = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 2

# Memory transfers policies
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3

# Error codes
ACL_ERROR_NONE = 0
