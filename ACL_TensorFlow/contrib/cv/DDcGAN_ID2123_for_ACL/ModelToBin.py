"""LICENSE"""
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

import cv2

#数据集路径
#data_path = sys.argv[1]
# 模型路径
#模型路径--图片路径
#output_path = sys.argv[2]

#image_path = data_path + '/export_dir'
output_path = './export_dir/bin'

label  = []
for i in range(3):
    index = i + 1

    ir_path = './test_imgs/20211209/1/IR/IR' + str(index) + '.bmp'
    vis_path = './test_imgs/20211209/1/VIS/VIS' + str(index) + '.bmp'

    ir_img = cv2.imread(ir_path, 0) / 255.0  # 读入数据
    vis_img = cv2.imread(vis_path, 0) / 255.0  # 读入数据
    ir_dimension = list(ir_img.shape)
    vis_dimension = list(vis_img.shape)

    ir_dimension.insert(0, 1)
    ir_dimension.append(1)
    vis_dimension.insert(0, 1)
    vis_dimension.append(1)
    print(ir_dimension[0])
    print(ir_dimension[1])
    print(ir_dimension[2])
    ir_img = ir_img.reshape(ir_dimension).astype('float32')
    vis_img = vis_img.reshape(vis_dimension).astype('float32')

    ir_img.tofile(output_path + "/IR/" + str(index) + ".bin") # 处理后的图片保存为bin文件
    vis_img.tofile(output_path + "/VIS/" + str(index) + ".bin")  # 处理后的图片保存为bin文件