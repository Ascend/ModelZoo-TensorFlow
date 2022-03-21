#!/usr/bin/env python3
#
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
#
# -*- coding: UTF-8 -*-

from npu_bridge.npu_init import *
import time
import roslibpy
import numpy as np


class RosTalker:
    """
    a ros-relative class. run as a ros topic talker.
    """
    def __init__(self, host, port=9090):
        print('Connecting to ROS master...')
        self.host = host
        self.port = port
        # ros connection
        self.client = roslibpy.Ros(host=self.host, port=self.port)
        self.talker = roslibpy.Topic(self.client, '/chatter', 'std_msgs/Float64MultiArray')
        self.client.run()
        count = 0
        while not self.client.is_connected:
            time.sleep(1)
            count += 1
            assert count < 5, 'ROS connection overtime'
        print('ROS connection established.')

    def send(self, angles):
        s0_l, s1_l, e0_l, e1_l, s0_r, s1_r, e0_r, e1_r = angles
        self.talker.publish(roslibpy.Message({'layout': {'dim': [{'label': 'right_arm', 'size': 3, 'stride': 1}],
                                                         'data_offset': 0},
                                              'data': [s0_l.astype(np.float64), s1_l.astype(np.float64),
                                                       e0_l.astype(np.float64), e1_l.astype(np.float64),
                                                       s0_r.astype(np.float64), s1_r.astype(np.float64),
                                                       e0_r.astype(np.float64), e1_r.astype(np.float64)]}))
        print('Sending message to ros master:', angles)
