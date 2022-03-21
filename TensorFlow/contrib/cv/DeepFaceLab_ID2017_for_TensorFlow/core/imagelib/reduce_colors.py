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
import cv2
from PIL import Image


def reduce_colors (img_bgr, n_colors):
    img_rgb = (img_bgr[...,::-1] * 255.0).astype(np.uint8)
    img_rgb_pil = Image.fromarray(img_rgb)
    img_rgb_pil_p = img_rgb_pil.convert('P', palette=Image.ADAPTIVE, colors=n_colors)

    img_rgb_p = img_rgb_pil_p.convert('RGB')
    img_bgr = cv2.cvtColor( np.array(img_rgb_p, dtype=np.float32) / 255.0, cv2.COLOR_RGB2BGR )

    return img_bgr
