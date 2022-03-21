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

# ### Init

# In[1]:

import os
import argparse
#os.environ['LD_PRELOAD'] = '/usr/lib64/libgomp.so.1:/usr/libexec/coreutils/libstdbuf.so'
#os.system('pip list')
#os.system('pip install keras==2.2.4')
#os.system('pip install munch==2.5.0')
#os.system('pip list')

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./dataset")
parser.add_argument("--model_path", type=str, default="./dataset")
parser.add_argument("--model_name", type=str, default="./dataset")
parser.add_argument("--pretrain_model", type=str, default="./dataset")
config = parser.parse_args()

# 数据集、特征、cvs文件、预训练的模型的模型
data_url = config.data_path
os.system('mkdir -p ~/.keras/models')
cp_shell = ("cp -r %s/* ~/.keras/models" % config.pretrain_model)
os.system(cp_shell)

from kutils import model_helper as mh
from kutils import applications as apps
from kutils import tensor_ops as ops
from kutils import generic as gen
from kutils import image_utils as img

import pandas as pd, numpy as np
from npu_bridge.npu_init import *

# 设置npu进行训练
ops.Startnpu()

dataset = data_url + 'metadata/AVA_data_official_test.csv'
images_path = data_url + 'dataset-mnist/'
ids = pd.read_csv(dataset)

files = os.listdir(images_path)
# 输出图片的个数
print("数据集里的文件个数Files number:", len(files))

# ### Load combined model

# In[ ]:


# load base model
model_file_name = config.model_name
input_shape = (None, None, 3)
model_base = apps.model_inceptionresnet_pooled(input_shape)
pre   = apps.process_input[apps.InceptionResNetV2]

# MODEL DEF
from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model

input_feats = Input(shape=(5,5,16928), dtype='float32')
x = apps.inception_block(input_feats, size=1024)
x = GlobalAveragePooling2D(name='final_GAP')(x)

pred = apps.fc_layers(x, name       = 'head',
                      fc_sizes      = [2048, 1024, 256,  1],
                      dropout_rates = [0.25, 0.25, 0.5, 0],
                      batch_norm    = 2)

model = Model(inputs  = input_feats, 
              outputs = pred)

gen_params = dict(batch_size    = 1,
                  data_url     = images_path,                  
                  process_fn    = pre,
                  input_shape   = input_shape,
                  inputs        = 'image_name',
                  outputs       = 'MOS', 
                  fixed_batches = False)

helper = mh.ModelHelper(model, model_file_name, ids, 
                        gen_params = gen_params)


# load head model
helper.load_model(model_name = data_url + config.model_path + config.model_name)

# join base and head models
helper.model = Model(inputs  = model_base.input, 
                     outputs = model(model_base.output))



# ### Predict score of a single image
# choose an image from the test set
image_path = images_path + ids[ids.set == 'test'].iloc[0].image_name
print(ids[ids.set == 'test'].iloc[0].image_name)


# load, pre-process it, and pass it to the model
I = pre(img.read_image(image_path))
I = np.expand_dims(I, 0)
I_score = helper.model.predict(I)
print('predicted image score:', I_score[0][0])

# # ### Test full model
# helper.gen_params.verbose = True
# _, _, SRCC_test, PLCC_test, ACC_test = apps.test_rating_model(helper, accuracy_thresh = 5,ids = ids[ids.set == 'test'])



# 关闭npu
ops.endnpu()

