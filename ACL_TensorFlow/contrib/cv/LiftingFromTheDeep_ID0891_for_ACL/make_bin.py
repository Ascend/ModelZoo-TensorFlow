# Copyright 2022 Huawei Technologies Co., Ltd
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

image = cv2.imread('./input/test_image.jpg')
image = cv2.resize(image, (654, 368))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
b_image = np.array(image[np.newaxis] / 255.0 - 0.5, dtype=np.float32)
b_image.tofile('./input/b_image.bin')
