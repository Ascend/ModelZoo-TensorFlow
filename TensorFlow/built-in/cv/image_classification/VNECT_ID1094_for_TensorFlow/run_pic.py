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
import cv2
from src import utils
from src.hog_box import HOGBox
from src.estimator import VNectEstimator


box_size = 368
hm_factor = 8
joints_num = 21
scales = [1.0, 0.7]
joint_parents = [1, 15, 1, 2, 3, 1, 5, 6, 14, 8,
                 9, 14, 11, 12, 14, 14, 1, 4, 7, 10,
                 13]
estimator = VNectEstimator()
img = cv2.imread('./pic/test_pic.jpg')
hog = HOGBox()
hog.clicked = True
choose, rect = hog(img.copy())
x, y, w, h = rect
img_cropped = img[y: y + h, x: x + w, :]
joints_2d, joints_3d = estimator(img_cropped)
# 3d plotting
utils.draw_limbs_3d(joints_3d, joint_parents)
# 2d plotting
joints_2d[:, 0] += y
joints_2d[:, 1] += x
img_draw = utils.draw_limbs_2d(img.copy(), joints_2d, joint_parents, [x, y, w, h])
cv2.imshow('2D Prediction', img_draw)

cv2.waitKey()
cv2.destroyAllWindows()
