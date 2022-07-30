# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
import cv2
pic = cv2.imread("C:/Users/Amber/Desktop/mono_test.jpg") 
pic = cv2.resize(pic1, (640, 192))
pic = np.reshape(pic1, (1, 192, 640, 3)) 
print(pic.shape)
print(pic.dtype)
pic.tofile("C:/Users/Amber/Desktop/mono_test.bin")