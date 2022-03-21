#!/usr/bin/env python
# coding: utf-8
#
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
#

# # Init

# npu 迁移之后训练所需添加的代码

import os
os.system('pip install keras==2.2.4')
os.system('pip install munch==2.5.0')
os.system('pip install pandas==1.2.4')
os.system('pip install scikit-image==0.16.2')
os.system('pip install h5py==2.8.0')
os.system('pip install scikit-learn==0.21.3')
import numpy as np
from kutils import applications as apps
from kutils import generic
from kutils import model_helper as mh
print('导入kutils成功')
import argparse
# import numpy as np
import pandas as pd
from npu_bridge.npu_init import *





# 数据集、特征、cvs文件的路径

data_path = '/home/ma-user/modelarts/inputs/data_url_0/'
files = os.listdir(data_path)
# 输出文件个数
print("数据集里的文件个数Files number:", len(files))

# 设置图片的路径
images_path = data_path + 'dataset-mnist/'

# 创建接收.cvs的文件目录
dataset = data_path + 'metadata/AVA_data_official_test.csv'
ids = pd.read_csv(dataset)

# ### Train on MLSP wide features
fc1_size = 2048
image_size = '[orig]'
input_size = (5,5,16928)

#创建特征解析的存储路径
features_root = data_path + 'features/irnv2_mlsp_wide_orig/'

input_shape = (None, None, 3)
# 输入的形状是3通道的，但是大小没定

# ### InceptionResNetV2 MLSP wide

model = apps.model_inceptionresnet_pooled(input_shape)
pre = apps.process_input[apps.InceptionResNetV2]
model_name = ''

# # Save features
# ### original sized images, no augmentation

gen_params = dict(batch_size=1,
                  data_path=images_path,
                  input_shape=('orig',),
                  inputs=('image_name',),
                  process_fn=pre,
                  fixed_batches=False)

helper = mh.ModelHelper(model, model_name, ids,
                        features_root=features_root,
                        gen_params=gen_params)

print('Saving features')
batch_size = 1024
numel = len(ids)
for i in range(0, numel, batch_size):
    istop = min(i + batch_size, numel)
    print('Processing images', i, ':', istop)
    ids_batch = ids[i:istop].reset_index(drop=True)
    helper.save_activations(ids=ids_batch, verbose=True, save_as_type=np.float16)

# check contents of saved file
with generic.H5Helper(features_root + 'i1[orig]_lfinal_o1[5,5,16928]_r1.h5', 'r') as h:
    print(h.summary())
