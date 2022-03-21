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
import auxiliaryFunctions as sf

# Set this according to your quantity of images
nImg = 1

# Call data pre-processing function
test_data = sf.getTestingData(nImg=nImg)
atb, mask = test_data['atb'], test_data['mask']

# Save inputs as binaries for the .om model. The destination is pwd by default
atb.astype(np.float32).tofile('atb.bin')
mask.astype(np.complex64).tofile('mask.bin')
