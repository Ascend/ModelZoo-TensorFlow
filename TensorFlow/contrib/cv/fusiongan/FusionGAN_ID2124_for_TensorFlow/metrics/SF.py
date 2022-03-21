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
import math
from PIL import Image

def spatialF(image):
    '''
    compute SF
    img:input image
    '''
    image = image.convert('L')
    image = np.array(image)
    M = image.shape[0]
    N = image.shape[1]
    # print(image)
    cf = 0
    rf = 0
    # for i in range(1, M - 1):
    #     for j in range(1, N - 1):
    #         dx = float(image[i, j - 1]) - float(image[i, j])
    #         rf += dx ** 2
    #         dy = float(image[i - 1, j]) - float(image[i, j])
    #         cf += dy ** 2
    for i in range(0, M):
        for j in range(1, N):
            dx = float(image[i, j - 1]) - float(image[i, j])
            rf += dx ** 2
    for i in range(1, M):
        for j in range(0, N):
            dy = float(image[i - 1, j]) - float(image[i, j])
            cf += dy ** 2
    # print(str(cf)+'\n'+str(rf))
    RF = math.sqrt(rf / (M * N))
    CF = math.sqrt(cf / (M * N))
    SF = math.sqrt(RF ** 2 + CF ** 2)
    # print(SF)
    return round(SF, 2)

# if __name__ == "__main__":
#     fusedimg_path = "./metrics/result/3.bmp"

 
#     print(spatialF(fusedimg_path))