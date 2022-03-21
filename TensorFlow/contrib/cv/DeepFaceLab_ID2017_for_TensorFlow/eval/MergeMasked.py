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

import sys
import traceback
import random

import cv2
import numpy as np

from core import imagelib
from core.cv2ex import *
from core.interact import interact as io
from facelib import FaceType, LandmarksProcessor

is_windows = sys.platform[0:3] == 'win'
xseg_input_size = 256


def MergeMaskedFace (predictor_func, predictor_input_shape,
                     cfg, img_bgr, img_face_landmarks):

    img_size = img_bgr.shape[1], img_bgr.shape[0]
    img_face_mask_a = LandmarksProcessor.get_image_hull_mask (img_bgr.shape, img_face_landmarks)

    input_size = predictor_input_shape[0]
    output_size = input_size

    face_mat = LandmarksProcessor.get_transform_mat(img_face_landmarks, output_size, face_type=cfg.face_type)
    dst_face_bgr = cv2.warpAffine( img_bgr, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
    dst_face_bgr = np.clip(dst_face_bgr, 0, 1)

    dst_face_mask_a_0 = cv2.warpAffine( img_face_mask_a, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC )
    dst_face_mask_a_0 = np.clip(dst_face_mask_a_0, 0, 1)

    predictor_input_bgr      = cv2.resize(dst_face_bgr, (input_size,input_size) )
    # name_index = random.randint(1, 999)
    # filename = "/root/DFL_saved_GPU/DeepFaceLab/workspace_2/temp/debug_"+str(name_index)+".png"
    # filename1 = "/root/DFL_saved_GPU/DeepFaceLab/workspace_2/temp/debug_"+str(name_index)+"_.png"
    # cv2_imwrite(filename, (predictor_input_bgr * 255).astype(np.uint8))
    # cv2_imwrite(filename1, (dst_face_mask_a_0 * 255).astype(np.uint8))
    # io.log_info(predictor_input_bgr.shape)
    # io.log_info(dst_face_mask_a_0.shape)

    predicted_ssim = predictor_func(face=predictor_input_bgr, target_dstm=dst_face_mask_a_0)
    return predicted_ssim


def MergeMasked (predictor_func,
                 predictor_input_shape,
                 cfg,
                 frame_info):
    img_bgr_uint8 = cv2_imread(frame_info.filepath)
    img_bgr_uint8 = imagelib.normalize_channels (img_bgr_uint8, 3)
    img_bgr = img_bgr_uint8.astype(np.float32) / 255.0

    ssim_result = []
    for face_num, img_landmarks in enumerate( frame_info.landmarks_list ):
        ssim = MergeMaskedFace(predictor_func, predictor_input_shape,
                               cfg, img_bgr, img_landmarks)
        ssim_result.append(ssim)

    return np.mean(ssim_result)
