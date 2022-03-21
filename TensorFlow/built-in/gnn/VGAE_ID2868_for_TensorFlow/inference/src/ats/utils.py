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
Utilities
"""

from ats.constants import ACL_ERROR_NONE


def check_status(message, status):
    """
    Check status code
    """
    if status != ACL_ERROR_NONE:
        raise Exception("{} Failed status={}"
                        .format(message, status))
