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
label = np.load("./label.npy")

files = ["./binout/20220524_162230"+"/"+i for i in os.listdir("/root/zwy/binout/20220524_162230")]

files.sort(key=lambda x: int(x.split("/")[-1].split("_")[0]))

pred = []
for i in tqdm(files):
    tmp = np.fromfile(i,dtype=np.float32).reshape(4,100)
    pred += tmp.argmax(axis=1).tolist()
print(pred[:100])
print(label[:100])
res = (np.array(pred) == label).mean()

print("推理结果为：",res)
