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
from npu_bridge.npu_init import *
from enum import Enum


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18


CocoPairs = [(1, 2),
             (1, 5),
             (2, 3),
             (3, 4),
             (5, 6),
             (6, 7),
             (1, 8),
             (8, 9),
             (9, 10),
             (1, 11),
             (11, 12),
             (12, 13),
             (1, 0),
             (0, 14),
             (14, 16),
             (0, 15),
             (15, 17),
             (2, 16),
             (5, 17)]   # = 19
CocoPairsRender = CocoPairs[:-2]


CocoPairsNetwork = [(12, 13),
                    (20, 21),
                    (14, 15),
                    (16, 17),
                    (22, 23),
                    (24, 25),
                    (0, 1),
                    (2, 3),
                    (4, 5),
                    (6, 7),
                    (8, 9),
                    (10, 11),
                    (28, 29),
                    (30, 31),
                    (34, 35),
                    (32, 33),
                    (36, 37),
                    (18, 19),
                    (26, 27)]  # = 19


CocoColors = [[0, 100, 255],
              [0, 100, 255],
              [0, 255, 255],
              [0, 100, 255],
              [0, 255, 255],
              [0, 100, 255],
              [0, 255, 0],
              [255, 200, 100],
              [255, 0, 255],
              [0, 255, 0],
              [255, 200, 100],
              [255, 0, 255],
              [0, 0, 255],
              [255, 0, 0],
              [200, 200, 0],
              [255, 0, 0],
              [200, 200, 0],
              [0, 0, 0]]


