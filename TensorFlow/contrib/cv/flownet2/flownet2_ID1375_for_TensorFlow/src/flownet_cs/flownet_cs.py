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
from npu_bridge.npu_init import *
from ..net import Net, Mode
from ..flownet_c.flownet_c import FlowNetC
from ..flownet_s.flownet_s import FlowNetS
# from ..flow_warp import flow_warp
from ..flow_warp import dense_image_warp
import tensorflow as tf


class FlowNetCS(Net):

    def __init__(self, mode=Mode.TRAIN, debug=False):
        self.net_c = FlowNetC(mode, debug)
        self.net_s = FlowNetS(mode, debug)
        super(FlowNetCS, self).__init__(mode=mode, debug=debug)

    def model(self, inputs, training_schedule, trainable=True):
        with tf.variable_scope('FlowNetCS'):
            # Forward pass through FlowNetC with weights frozen
            net_c_predictions = self.net_c.model(inputs, training_schedule, trainable=False)

            # Perform flow warping (to move image B closer to image A based on flow prediction)
            # warped = flow_warp(inputs['input_b'], net_c_predictions['flow'])
            warped = dense_image_warp(inputs['input_b'], net_c_predictions['flow'])

            # Compute brightness error: sqrt(sum (input_a - warped)^2 over channels)
            brightness_error = inputs['input_a'] - warped
            brightness_error = tf.square(brightness_error)
            brightness_error = tf.reduce_sum(brightness_error, keepdims=True, axis=3)
            brightness_error = tf.sqrt(brightness_error)

            # Gather all inputs to FlowNetS
            inputs_to_s = {
                'input_a': inputs['input_a'],
                'input_b': inputs['input_b'],
                'warped': warped,
                'flow': net_c_predictions['flow'] * 0.05,
                'brightness_error': brightness_error,
            }

            return self.net_s.model(inputs_to_s, training_schedule, trainable=trainable)

    def loss(self, flow, predictions):
        return self.net_s.loss(flow, predictions)

