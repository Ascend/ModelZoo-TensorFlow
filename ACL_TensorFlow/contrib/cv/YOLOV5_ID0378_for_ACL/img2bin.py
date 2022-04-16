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
import os
from tqdm import tqdm
from PIL import Image
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input, resize_image)
from utils.utils_bbox import DecodeBox

VOCdevkit_path  = 'VOCdevkit'
image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

# for image_id in tqdm(image_ids):
image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+"000001"+".jpg")
image       = Image.open(image_path)
image       = cvtColor(image)
image_data  = resize_image(image, (640, 640), True)
image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
image_data.tofile('bin/image.bin')