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
# ===========================
#   Author      : ChenZhou
#   Time        : 2021/11
#   Language    : Python
# ===========================
import os
import numpy as np
files = 'C:\\Users\\ChenZhou\\Desktop\\jiaoyan'
output_num = 0
out = os.listdir(files)
acc = 0
biao=['person','dog','elephant','giraffe','guitar','horse','house']
for file in out:
    print(file)
    check=0;
    for i in biao:
        if file==i:
            break
        check+=1
    print(check)
    out2 = os.listdir(files+'/'+file)
    for img in (out2):
        print(img)
        output_num += 1
        tmp = np.fromfile(files+'/'+file+'/'+img, dtype='float32')
        print(tmp)
        inf_label = int(np.argmax(tmp))
        #print(inf_label)
        if inf_label==check:
            acc=acc+1

print(acc)
print(output_num)
print(acc/output_num)
