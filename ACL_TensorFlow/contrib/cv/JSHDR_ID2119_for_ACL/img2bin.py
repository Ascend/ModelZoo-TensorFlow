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
import pandas as pd
import cv2

images_path = 'D:/Program/code/test_image/'
image=cv2.imread(images_path+'smccc.png')
image = cv2.resize(image,(64,64))

image = np.array(image)
image = image.astype(np.float32)
image = image/255
#image = np.expand_dims(image, axis=0)
print(image)
print(image.shape)
image.tofile(images_path+'4.bin')
'''
array_bytes = image.tobytes()
print(image.shape)
print(type(array_bytes))
print(array_bytes.shape)

with open('D:/Program/code/test_image/'+'2.bin','wb') as f:
    f.write(array_bytes)
'''