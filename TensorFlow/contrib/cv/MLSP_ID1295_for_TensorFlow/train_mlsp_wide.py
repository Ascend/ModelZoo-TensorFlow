##!/usr/bin/env python
## coding: utf-8
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
#
#os.system('pip list')
#os.system('pip install keras==2.2.4')
#os.system('pip install munch==2.5.0')
#os.system('pip install pandas==1.2.4')
#os.system('pip install scikit-image==0.16.2')
#os.system('pip install h5py==2.8.0')
#os.system('pip install scikit-learn==0.21.3')
#os.system('pip list')
#print('ls：')
#os.system('ls')
#print('pwd：')
#os.system('pwd')
#
import os
from kutils import applications as apps
from kutils import model_helper as mh
from kutils import tensor_ops as ops
import argparse
import pandas as pd
from npu_bridge.npu_init import *
import time

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./dataset")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--end_step", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-4, help="lr初始化为le-4,epoch每20个除以10")
config = parser.parse_args()

# 设置npu进行训练
ops.Startnpu()

# 数据集、特征、cvs文件的路径
data_path = config.data_path

print("判断features是否存在：")
features_file = data_path + '/features/irnv2_mlsp_wide_orig/i1[orig]_lfinal_o1[5,5,16928]_r1.h5'
print(os.path.exists(features_file))

files = os.listdir(data_path)
# 输出文件个数
print("数据集里的文件个数Files number:", len(files))

# 创建接收.cvs的文件目录
dataset = data_path + '/metadata/AVA_data_official_test.csv'
ids = pd.read_csv(dataset)

# ### Train on MLSP wide features
fc1_size = 2048
image_size = '[orig]'
input_size = (5,5,16928)

# 创建存放model的路径
output_path = data_path

# model_name = irnv2_mlsp_wide_orig
model_name = 'irnv2_mlsp_wide_orig'


loss = 'MSE'
bn = 2
fc_sizes = [fc1_size, fc1_size/2, fc1_size/8,  1]
dropout_rates = [0.25, 0.25, 0.5, 0]

monitor_metric = 'val_plcc_tf'; monitor_mode = 'max'
metrics = ["MAE", ops.plcc_tf]
outputs = 'MOS'

# MODEL DEF
from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model

input_feats = Input(shape=input_size, dtype='float32')

# SINGLE-block
x = apps.inception_block(input_feats, size=1024)
x = GlobalAveragePooling2D(name='final_GAP')(x)

pred = apps.fc_layers(x, name       = 'head',
                 fc_sizes      = fc_sizes,
                 dropout_rates = dropout_rates,
                 batch_norm    = bn)

model = Model(inputs=input_feats, outputs=pred)
gen_params = dict(batch_size    = 128,
                  data_path     = features_file,                  
                  input_shape   = input_size,
                  inputs        = 'image_name',
                  outputs       = outputs, 
                  random_group  = False,
                  fixed_batches = True)
# 调用dict()函数创建gen_params字典
# mh就是model_helper
helper = mh.ModelHelper(model, model_name, ids, 
                     max_queue_size = 128,
                     loss           = loss,
                     metrics        = metrics,
                     monitor_metric = monitor_metric, 
                     monitor_mode   = monitor_mode,
                     multiproc      = False, workers = 1,
#                    multiproc      = True, workers = 3,
                     early_stop_patience = 5,
                     logs_root      = output_path + 'logs',
                     models_root    = output_path + 'models',
                     gen_params     = gen_params)

helper.model_name.update(fc1 = '[%d]' % fc1_size, 
                         im  = image_size,
                         bn  = bn,
                         do  = str(dropout_rates).replace(' ',''),
                         mon = '[%s]' % monitor_metric,
                         ds  = '[%s]' % os.path.split(dataset)[1])
print('打印模型的名字：')
print (helper.model_name())


# In[4]:

#训练
start_time = time.time()
helper.load_model()
helper.train(lr=config.lr, epochs=config.epochs, end_step=config.end_step)
duration = time.time() - start_time
print("duration_per_train_func is %.3f seconds" % duration)

if helper.load_model():
    y_test, y_pred, SRCC_test, PLCC_test, ACC_test = apps.test_rating_model(helper, accuracy_thresh=5)



# 关闭npu
ops.endnpu()
