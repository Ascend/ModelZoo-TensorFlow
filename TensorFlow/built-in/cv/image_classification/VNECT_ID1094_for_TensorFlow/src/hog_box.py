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
import numpy as np


class HOGBox:
    """
    a simple HOG-method-based human tracking box
    """
    # mouse click flag
    clicked = False

    def __init__(self):
        print('Initializing HOGBox...')
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._box_init_window_name = 'Click mouse to initialize bounding box'
        cv2.namedWindow(self._box_init_window_name)
        cv2.setMouseCallback(self._box_init_window_name, self.on_mouse)
        print('HOGBox initialized.')

    def __call__(self, img):
        H, W = img.shape[:2]
        found, w = self.hog.detectMultiScale(img)
        rect = self.cal_rect(found[np.argmax([found[i, 2] * found[i, 3] for i in range(len(found))])], H, W) \
            if len(found) else [0, 0, W, H]  # biggest area
        # rect = self.cal_rect(found[np.argmax(w)], H, W) if len(found) else [0, 0, W, H]  # biggest weight
        self.draw_rect(img, rect)
        scale = 400 / H
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        cv2.imshow(self._box_init_window_name, img)

        if self.clicked:
            cv2.destroyWindow(self._box_init_window_name)
        return self.clicked, rect

    def on_mouse(self, event, x, y, flags, param):
        """
        attain mouse clicking message
        """
        if event == cv2.EVENT_LBUTTONUP:
            self.clicked = True

    @staticmethod
    def cal_rect(rect, H, W):
        """
        calculate the box size and position
        """
        x, y, w, h = rect
        offset_w = int(0.4 / 2 * W)
        offset_h = int(0.2 / 2 * H)
        return [np.max([x - offset_w, 0]),  # x
                np.max([y - offset_h, 0]),  # y
                np.min([x + w + offset_w, W]) - np.max([x - offset_w, 0]),  # w
                np.min([y + h + offset_h, H]) - np.max([y - offset_h, 0])]  # h

    @staticmethod
    def draw_rect(img, rect):
        """
        draw bounding box in the BB initialization window, and record current rect (x, y, w, h)
        """
        x, y, w, h = rect
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(img, pt1, pt2, (60, 66, 207), 4)
