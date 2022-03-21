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

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import os
import scipy.io
import math
# os.system('cd /home/nanshen/xutan/yixin/circlesoftmax')
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--graphpath',default='/home/nanshen/xutan/yixin/circlesoftmax/logs/b16_epoch10_lr5en2/ckpt/5656/softmaxcircleloss.ckpt-5656.meta', type=str)
parser.add_argument('--parampath',default='/home/nanshen/xutan/yixin/circlesoftmax/logs/b16_epoch10_lr5en2/ckpt/5656', type=str)
parser.add_argument('--logdir',default='/home/nanshen/xutan/yixin/circlesoftmax/logs/b16_epoch10_lr5en2', type=str)
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--test_dir',default='Data/Market-1501-v15.09.15/pytorch',type=str, help='./test_data')
parser.add_argument('--savepath', default='checkpoint', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()

opt.nclasses = 751 

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
# name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
# if len(gpu_ids)>0:
#     torch.cuda.set_device(1)
#     cudnn.benchmark = True


### tf model #####
import os
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_ids
import tensorflow as tf

sess=tf.Session()
saver = tf.train.import_meta_graph(opt.graphpath)
saver.restore(sess, tf.train.latest_checkpoint(opt.parampath))

# 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
graph = tf.get_default_graph()

x = graph.get_tensor_by_name("inputs:0")
keep_rate = graph.get_tensor_by_name("keep_rate:0")
# y = graph.get_tensor_by_name("labels:0")
istrain = graph.get_tensor_by_name("is_training:0")
features = graph.get_tensor_by_name("Logits/features:0")
def extract_feature_tf(batch_im):
    return sess.run(features,feed_dict={x:batch_im,istrain:False,keep_rate:1.0})

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
h, w = 256, 128
    

data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
############### Ten Crop        
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop) 
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
])

data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
# use_gpu = torch.cuda.is_available()



######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(dataloaders):
    features = torch.FloatTensor()
    count = 0
    from tqdm import tqdm
    for data in tqdm(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n
        # print(count)
        ff = torch.FloatTensor(n,512).zero_()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            # input_img = Variable(img.cuda())
            input_img = img.cpu().numpy()
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(torch.tensor(input_img), scale_factor=scale, mode='bicubic', align_corners=False)
                    input_img = input_img.cpu().numpy()
                # outputs = extract_feature_tf(input_img)
                input_img = input_img.transpose(0,2,3,1)
                outputs = extract_feature_tf(input_img)
                outputs = Variable(torch.from_numpy(outputs))
                ff += outputs
        features = torch.cat((features,ff.data.cpu()), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

if opt.multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam,mquery_label = get_id(mquery_path)


# Extract feature
gallery_feature = extract_feature(dataloaders['gallery'])
query_feature = extract_feature(dataloaders['query'])
if opt.multi:
    mquery_feature = extract_feature(dataloaders['multi-query'])

import time
timestamp = int(time.time())

# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat(f'res/pytorch_result_{timestamp}.mat',result)

# print(opt.name)
# result = 'checkpoint/result.txt'
os.system(f'python code/eval.py --filename res/pytorch_result_{timestamp}.mat --logdir {opt.logdir}')

if opt.multi:
    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
    scipy.io.savemat(opt.savepath+'multi_query.mat',result)
