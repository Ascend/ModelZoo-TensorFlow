"""
to bin
"""
# coding=utf-8
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

import numpy as np
from models.pre_input import get_right_images

feature_tst, label_tst, mask_tst = get_right_images('E:/HuaweiProject/TF_re/Cascade_re/data/' + 'chest_test_acc3.hdf5')

for i in range(feature_tst.shape[0]):
    feature = feature_tst[i, :, :, :].astype('float32')
    mask = mask_tst[i, :, :].astype('float32')
    label = label_tst[i, :, :, :].astype('float32')
    feature.tofile('test_bin/feature/{}.bin'.format(i))
    mask.tofile('test_bin/mask/{}.bin'.format(i))
    label.tofile('test_bin/label/{}.bin'.format(i))

print('ok')
