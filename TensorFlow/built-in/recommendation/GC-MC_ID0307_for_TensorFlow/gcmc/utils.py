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
from __future__ import division
from __future__ import print_function


def construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero, v_features_nonzero,
                        support, support_t, labels, u_indices, v_indices, class_values,
                        dropout, u_features_side=None, v_features_side=None):
    """
    Function that creates feed dictionary when running tensorflow sessions.
    """

    feed_dict = dict()
    feed_dict.update({placeholders['u_features']: u_features})
    feed_dict.update({placeholders['v_features']: v_features})
    feed_dict.update({placeholders['u_features_nonzero']: u_features_nonzero})
    feed_dict.update({placeholders['v_features_nonzero']: v_features_nonzero})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['support_t']: support_t})

    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['user_indices']: u_indices})
    feed_dict.update({placeholders['item_indices']: v_indices})

    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['class_values']: class_values})

    if (u_features_side is not None) and (v_features_side is not None):
        feed_dict.update({placeholders['u_features_side']: u_features_side})
        feed_dict.update({placeholders['v_features_side']: v_features_side})

    return feed_dict
