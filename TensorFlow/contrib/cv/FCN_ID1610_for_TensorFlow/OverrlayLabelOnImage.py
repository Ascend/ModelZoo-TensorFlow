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
###################Overlay Label on image Mark label on  the image in transperent form###################################################################################
def OverLayLabelOnImage(ImgIn,Label,W):
    #ImageIn is the image
    #Label is the label per pixel
    # W is the relative weight in which the labels will be marked on the image
    # Return image with labels marked over it
    Img=ImgIn.copy()
    TR = [0,1, 0, 0,   0, 1, 1, 0, 0,   0.5, 0.7, 0.3, 0.5, 1,    0.5]
    TB = [0,0, 1, 0,   1, 0, 1, 0, 0.5, 0,   0.2, 0.2, 0.7, 0.5,  0.5]
    TG = [0,0, 0, 0.5, 1, 1, 0, 1, 0.7, 0.4, 0.7, 0.2, 0,   0.25, 0.5]
    R = Img[:, :, 0].copy()
    G = Img[:, :, 1].copy()
    B = Img[:, :, 2].copy()
    for i in range(1, Label.max()+1):
        if i<len(TR): #Load color from Table
           R[Label == i] = TR[i] * 255
           G[Label == i] = TG[i] * 255
           B[Label == i] = TB[i] * 255
        else: #Generate random label color
           R[Label == i] = np.mod(i*i+4*i+5,255)
           G[Label == i] = np.mod(i*10,255)
           B[Label == i] = np.mod(i*i*i+7*i*i+3*i+30,255)
    Img[:, :, 0] = Img[:, :, 0] * (1 - W) + R * W
    Img[:, :, 1] = Img[:, :, 1] * (1 - W) + G * W
    Img[:, :, 2] = Img[:, :, 2] * (1 - W) + B * W
    return Img