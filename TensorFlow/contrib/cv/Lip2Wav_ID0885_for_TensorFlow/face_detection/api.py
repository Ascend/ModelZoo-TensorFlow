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


from __future__ import print_function
from npu_bridge.npu_init import *
import os
import torch
from torch.utils.model_zoo import load_url
from enum import Enum
import numpy as np
import cv2
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
from .utils import *


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value

ROOT = os.path.dirname(os.path.abspath(__file__))

class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False, face_detector='sfd', verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        face_detector_module = __import__('face_detection.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)

    def get_detections_for_batch(self, images):
        images = images[..., ::-1]
        detected_faces = self.face_detector.detect_from_batch(images.copy())
        results = []

        for i, d in enumerate(detected_faces):
            if len(d) == 0:
                results.append(None)
                continue
            d = d[0]
            d = np.clip(d, 0, None)
            
            x1, y1, x2, y2 = map(int, d[:-1])

            inp = images[i][y1 : y2, x1 : x2, ::-1]
            results.append((inp, x1, y1, x2, y2))

        return results
