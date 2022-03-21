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

from enum import IntEnum


class FaceType(IntEnum):
    # enumerating in order "next contains prev"
    HALF = 0
    MID_FULL = 1
    FULL = 2
    FULL_NO_ALIGN = 3
    WHOLE_FACE = 4
    HEAD = 10
    HEAD_NO_ALIGN = 20

    MARK_ONLY = 100, # no align at all, just embedded faceinfo

    @staticmethod
    def fromString (s):
        # s = full_face
        r = from_string_dict.get(s.lower())
        # r = FaceType.FULL
        if r is None:
            raise Exception('FaceType.fromString value error')
        return r

    @staticmethod
    def toString (face_type):
        return to_string_dict[face_type]


to_string_dict = {FaceType.HALF: 'half_face',
                  FaceType.MID_FULL: 'midfull_face',
                  FaceType.FULL: 'full_face',
                  FaceType.FULL_NO_ALIGN: 'full_face_no_align',
                  FaceType.WHOLE_FACE: 'whole_face',
                  FaceType.HEAD: 'head',
                  FaceType.HEAD_NO_ALIGN: 'head_no_align',
                  FaceType.MARK_ONLY: 'mark_only',
                 }
from_string_dict = {to_string_dict[x]: x for x in to_string_dict.keys()}