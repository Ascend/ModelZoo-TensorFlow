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
import tensorflow as tf

OFFSETS_OUTPUT_SIZE = {
    'box_3d': 6,
    'box_8c': 24,
    'box_8co': 24,
    'box_4c': 10,
    'box_4ca': 10,
}

ANG_VECS_OUTPUT_SIZE = {
    'box_3d': 2,
    'box_8c': 0,
    'box_8co': 0,
    'box_4c': 0,
    'box_4ca': 2,
}


def feature_fusion(fusion_method, inputs, input_weights):
    """Applies feature fusion to multiple inputs

    Args:
        fusion_method: 'mean' or 'concat'
        inputs: Input tensors of shape (batch_size, width, height, depth)
            If fusion_method is 'mean', inputs must have same dimensions.
            If fusion_method is 'concat', width and height must be the same.
        input_weights: Weight of each input if using 'mean' fusion method

    Returns:
        fused_features: Features after fusion
    """

    # Feature map fusion
    with tf.variable_scope('fusion'):
        fused_features = None

        if fusion_method == 'mean':
            rois_sum = tf.reduce_sum(inputs, axis=0)
            rois_mean = tf.divide(rois_sum, tf.reduce_sum(input_weights))
            fused_features = rois_mean

        elif fusion_method == 'concat':
            # Concatenate along last axis
            last_axis = len(inputs[0].get_shape()) - 1
            fused_features = tf.concat(inputs, axis=last_axis)

        elif fusion_method == 'max':
            fused_features = tf.maximum(inputs[0], inputs[1])

        else:
            raise ValueError('Invalid fusion method', fusion_method)

    return fused_features

