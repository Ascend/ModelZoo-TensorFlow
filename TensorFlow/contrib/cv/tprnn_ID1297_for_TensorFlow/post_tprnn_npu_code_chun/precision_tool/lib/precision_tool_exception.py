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

# coding=utf-8
import logging


class PrecisionToolException(Exception):
    """
    Class for PrecisionTool Exception
    """

    def __init__(self, error_info):
        super(PrecisionToolException, self).__init__()
        self.error_info = error_info


def catch_tool_exception(fuc):
    def handle(*args, **kwargs):
        log = logging.getLogger()
        try:
            return fuc(*args, **kwargs)
        except PrecisionToolException as pte:
            log.warning(pte.error_info)
        except SystemExit:
            # do not exit
            log.debug("Exit")

    return handle
