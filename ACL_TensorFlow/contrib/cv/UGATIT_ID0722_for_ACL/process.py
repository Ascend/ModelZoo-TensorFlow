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
import imageio


fake = np.fromfile('./output/BtoA/20220509_212345/UGATIT_BtoA_output_0.bin',np.float32).reshape(256,256,3)
fake = ((fake+1)/2)*255
imageio.imsave('./result/BtoA/0000.png',fake)
