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

bin_path = '~/Polygen/bin/vertex_model_output_0.bin'  # 指定要转换的bin文件 即vertex model推理生成的bin文件
vertex = np.fromfile(bin_path, dtype="float32")
vertex = vertex.reshape(2, 400, 4)
vertex[..., :3].tofile('~/Polygen/bin/f_vertex.bin')
np.squeeze(vertex[..., 3:], axis=2).tofile('~/Polygen/bin/f_mask.bin')
