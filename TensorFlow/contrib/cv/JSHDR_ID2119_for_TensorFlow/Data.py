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

from npu_bridge.npu_init import *
import os
import xml.etree.ElementTree as ET
import cv2
#import config as cfg
import codecs
#import selectivesearch
import numpy as np
import math
import sys
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt

import pandas as pd
import random

def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):

    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img


class data(object):

    def __init__(self, is_save=True):
        #self.train_list = cfg.Train_list   #Train_list = r'./Data/train_list.txt'

        #self.images_path = cfg.Images_path              #Images_path = r'./Data/Images/'
        #self.processed_path = cfg.Processed_path  Processed_path = r'./Data/Processed'

        self.image_w = 1280
        self.image_h = 720

        self.batch_size = 1#4#32#cfg.Batch_size   #Batch_size = 1
        self.train_images_index = []
        self.cursor = 0
        self.epoch = 0

    def get_batch(self,dataset,data_path):
        if len(self.train_images_index) == 0:
            dataset = dataset
            ids = pd.read_csv(dataset)
            #print('load train list----------')
            for line in ids.data:
                self.train_images_index.append(line)
            #np.random.shuffle(self.train_images_index) i need't this random
        images = []
        rois = []
        label = []
        #print(len(self.train_images_index))
        for i in range(self.batch_size):
            #images_path = os.path.join(self.images_path, self.train_images_index[self.cursor] + '.jpg')
            images_path = self.train_images_index[self.cursor]
            image = cv2.imread(data_path+images_path)
            images.append(image)
            #labels = np.load(os.path.join(self.processed_path, self.train_images_index[self.cursor] + '_data.npy'))
            #labels = sorted(labels.tolist(), reverse=True)
            #select_num = min(cfg.Roi_num, len(labels))
            #for rois_label in labels[0:select_num]:
            #    rois.append(
            #            [rois_label[1] + i, int(rois_label[2])-1, int(rois_label[3])-1, int(rois_label[4])+1, int(rois_label[5])+1])
            #    label.append((rois_label[6:]))
            self.cursor += 1
            if self.cursor >= len(self.train_images_index):
                self.cursor = 0
                self.epoch += 1
                #np.random.shuffle(self.train_images_index)  i need't this random
        #rois = np.array(rois)
        #label = np.array(label)
        images = np.array(images)
        return images#, rois, label

def get_patch(*args, patch_size=96, scale=1, multi_scale=False):
    ih, iw = args[0].shape[:2]
    ih, iw = 200#args[0].shape[:2]

    #print("ih = ")
    #print(ih)
    #print("iw = ")
    #print(iw)
    #print(patch_size)
    #print("this is patch_size")

    #kkk = input()
    #p = scale if multi_scale else 1
    #tp = p * patch_size
    #ip = tp // scale
    patch_size = 64
    tp = patch_size
    ip = patch_size


    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    #tx, ty = scale * ix, scale * iy
    tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    #print(ret[0].shape)
    #print(ret[2].shape) #list index out of range
    #print(len(ret))# there have two ret
    #print("yjjjjjjjjjjjjjyjjjjjjjjjjjj")
    #huu = input()
    return ret

def my_get_patch(image, patch_size=96, scale=1, multi_scale=False):
    #ih, iw = args[0].shape[:2]
    ih, iw = 200,200#args[0].shape[:2]

    #print("ih = ")
    #print(ih)
    #print("iw = ")
    #print(iw)
    #print(patch_size)
    #print("this is patch_size")

    #kkk = input()
    #p = scale if multi_scale else 1
    #tp = p * patch_size
    #ip = tp // scale
    patch_size = 64
    tp = patch_size
    ip = patch_size


    #ix = random.randrange(0, iw - ip + 1)
    #iy = random.randrange(0, ih - ip + 1)
    ix = 0
    iy = 0

    #tx, ty = scale * ix, scale * iy
    tx, ty = ix, iy

    ret = [
        image[iy:iy + ip, ix:ix + ip, :]
    ]

    #print(ret[0].shape)
    #print(ret[2].shape) #list index out of range
    #print(len(ret))# there have two ret
    #print("yjjjjjjjjjjjjjyjjjjjjjjjjjj")
    #huu = input()
    return ret

def my_get_patch_9block(image, patch_size=96, scale=1, multi_scale=False):
    #ih, iw = args[0].shape[:2]
    ih, iw = 200,200#args[0].shape[:2]
    h_size, w_size = 192,192#args[0].shape[:2]

    x = resize_image(image, h_size, w_size, out_image=None, resize_mode=cv2.INTER_CUBIC)

    patch_size = 64
    '''
    lr_list = [
        x[:, 0:patch_size, 0:patch_size, :],
        x[:, 0:patch_size, patch_size:2*patch_size, :],
        x[:, 0:patch_size, 2*patch_size:3*patch_size, :],
        x[:, patch_size:2*patch_size, 0:patch_size, :],
        x[:, patch_size:2*patch_size, patch_size:2*patch_size, :],
        x[:, patch_size:2*patch_size, 2*patch_size:3*patch_size, :],
        x[:, 2*patch_size:3*patch_size, 0:patch_size, :],
        x[:, 2*patch_size:3*patch_size, patch_size:2*patch_size, :],
        x[:, 2*patch_size:3*patch_size, 2*patch_size:3*patch_size, :]]
    '''
    lr_list = [
        x[0:patch_size, 0:patch_size, :],
        x[0:patch_size, patch_size:2*patch_size, :],
        x[0:patch_size, 2*patch_size:3*patch_size, :],
        x[patch_size:2*patch_size, 0:patch_size, :],
        x[patch_size:2*patch_size, patch_size:2*patch_size, :],
        x[patch_size:2*patch_size, 2*patch_size:3*patch_size, :],
        x[2*patch_size:3*patch_size, 0:patch_size, :],
        x[2*patch_size:3*patch_size, patch_size:2*patch_size, :],
        x[2*patch_size:3*patch_size, 2*patch_size:3*patch_size, :]]
    print("lr_list shape")
    print(lr_list[0].shape)#(8,3,64,64)


    sr_list = []
    for i in range(0, 3):
        lr_batch1 = np.expand_dims(lr_list[i*3], axis=0)
        lr_batch2 = np.expand_dims(lr_list[i*3+1], axis=0)
        lr_batch3 = np.expand_dims(lr_list[i*3+2], axis=0)
        lr_batch = np.concatenate((lr_batch1,lr_batch2,lr_batch3) , axis=0)
        #print(type(lr_batch))
        print(lr_batch.shape)
        #lr_batch.chunk(n_GPUs, dim=0)
        sr_batch = self.model(lr_batch)
        #print(type(sr_batch[0]))#<class 'torch.Tensor'>
        #print(len(sr_batch))#4
        #print(len(lr_batch))
        print(sr_batch.shape)#torch.Size([8, 3, 42, 42])
        #sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        sr_list.extend(sr_batch)

    output = x.new(h_size, w_size,c)

    output[0:patch_size, 0:patch_size, :]                               = sr_list[0][0:patch_size, 0:patch_size, :] 
    output[0:patch_size, patch_size:2*patch_size, :]                    = sr_list[1][0:patch_size, 0:patch_size, :] 
    output[0:patch_size, 2*patch_size:3*patch_size, :]                  = sr_list[2][0:patch_size, 0:patch_size, :] 
    output[patch_size:2*patch_size, 0:patch_size, :]                    = sr_list[3][0:patch_size, 0:patch_size, :] 
    output[patch_size:2*patch_size, patch_size:2*patch_size, :]         = sr_list[4][0:patch_size, 0:patch_size, :] 
    output[patch_size:2*patch_size, 2*patch_size:3*patch_size, :]       = sr_list[5][0:patch_size, 0:patch_size, :] 
    output[2*patch_size:3*patch_size, 0:patch_size, :]                  = sr_list[6][0:patch_size, 0:patch_size, :] 
    output[2*patch_size:3*patch_size, patch_size:2*patch_size, :]       = sr_list[7][0:patch_size, 0:patch_size, :] 
    output[2*patch_size:3*patch_size, 2*patch_size:3*patch_size, :]     = sr_list[8][0:patch_size, 0:patch_size, :] 


    return output

'''get_patch
    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr, hr = common.get_patch(
                lr,
                hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi_scale=multi_scale
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih, 0:iw]
            #hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr
'''

'''
    def train(self):
        for step in range(1, self.max_iter + 1):
            images, rois, labels = self.data.get_batch()
            feed_dict = {self.net.images: images, self.net.rois: rois, self.net.labels: labels}
            if step % self.summary_iter == 0:

                summary, loss, _ = self.sess.run([self.summary_op, self.net.total_loss, self.train_op],
                                                 feed_dict=feed_dict)
                self.writer.add_summary(summary, step)
                print("Data_epoch:" + str(self.data.epoch) + " " * 5 + "training_step:" + str(
                    step) + " " * 5 + "batch_loss:" + str(loss))

            else:
                self.sess.run([self.train_op], feed_dict=feed_dict)
            if step % self.save_iter == 0:
                print("saving the model as : " + self.ckpt_file)
                self.saver.save(self.sess, self.ckpt_file, global_step=self.global_step)
                
'''




