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

import argparse
arg = argparse.ArgumentParser()
import time 
timestamp = int(time.time())
arg.add_argument('-e','--expriment_name',type=str,default=f'test_{timestamp}')
arg.add_argument('-g','--gpu',type=str,default='0,1')
arg.add_argument('-w','--warmup',type=int,default=5)
arg.add_argument('-b','--batch_size',type=int,default=32)
arg.add_argument('-epoch','--epoch_size',type=int,default=120)
arg.add_argument('-modelname','--modelname',type=str,default='softmaxcircleloss')
arg.add_argument('-lr','--init_lr',type=float,default=0.02)
arg.add_argument('-lamda','--lamda',type=float,default=1)
arg.add_argument('-drop','--droprate',type=float,default=0.5)

# for huawei npu
arg.add_argument("--train_url", type=str, default="./output")
arg.add_argument("--data_url", type=str, default="./dataset")
arg.add_argument("--ckpt_url", type=str, default="./ckpt")

args = arg.parse_args()
# args.expriment_name = args.expriment_name+"_"+str(timestamp)
homedir = "/home/nanshen/xutan/yixin/circlesoftmax"
log_dir = f'{homedir}/logs/{args.expriment_name}/'
im_size = [256, 128]
market_dir = 'Data/Market-1501-v15.09.15'
# P=4
warmup = args.warmup
# K=4
m=0.25
gamma=32
lamda = args.lamda
num_classes=751
epoch_size = args.epoch_size
batch_size = args.batch_size
embedding_size=128
resnet_model_path = '/home/nanshen/xutan/yixin/resnet_v1_50.ckpt'  # Path to the pretrained model
model_save_path = log_dir + args.modelname  # Path to the model.ckpt-(num_steps) will be saved
num_bottleneck = 512
import os
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
init_lr = args.init_lr

