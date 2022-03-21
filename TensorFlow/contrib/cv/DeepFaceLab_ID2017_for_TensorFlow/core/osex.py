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

import os
import sys

if sys.platform[0:3] == 'win':
    from ctypes import windll
    from ctypes import wintypes

def set_process_lowest_prio():
    try:
        if sys.platform[0:3] == 'win':
            GetCurrentProcess = windll.kernel32.GetCurrentProcess
            GetCurrentProcess.restype = wintypes.HANDLE
            SetPriorityClass = windll.kernel32.SetPriorityClass
            SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            SetPriorityClass ( GetCurrentProcess(), 0x00000040 )
        elif 'darwin' in sys.platform:
            os.nice(10)
        elif 'linux' in sys.platform:
            os.nice(20)
    except:
        print("Unable to set lowest process priority")

def set_process_dpi_aware():
    if sys.platform[0:3] == 'win':
        windll.user32.SetProcessDPIAware(True)

def get_screen_size():
    if sys.platform[0:3] == 'win':
        user32 = windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    elif 'darwin' in sys.platform:
        pass
    elif 'linux' in sys.platform:
        pass
        
    return (1366, 768)
        