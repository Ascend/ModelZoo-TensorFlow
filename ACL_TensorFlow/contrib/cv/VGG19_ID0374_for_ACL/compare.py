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
import os
import numpy as np
OUTPUT = "/home/HwHiAiUser/omfile/out/image_bin"
files = os.listdir(OUTPUT)
label_dict = {}
file = open('val.txt', 'r')
for line in file.readlines():
    line = line.strip()
    k = line.split(' ')[0]
    v = int(line.split(' ')[1])
    label_dict[k] = v
file.close()
output_num = 0
check_num = 0
for file in files:
    if file.endswith(".bin"):
        tmp = np.fromfile(OUTPUT+'/'+file, dtype = 'float32')
        inf_label = int(np.argmax(tmp))
        output_num += 1
        pic_name = str(file.split(".JPEG")[0])+".JPEG"
        print("%s, inference label:%d, gt_label:%d" % (pic_name, inf_label, label_dict[pic_name]))
        if inf_label == label_dict[pic_name]:
          check_num += 1
top1_accuarcy = check_num / output_num
print("Totol pic num: %d, Top1 accuarcy: %.4f" % (output_num, top1_accuarcy))