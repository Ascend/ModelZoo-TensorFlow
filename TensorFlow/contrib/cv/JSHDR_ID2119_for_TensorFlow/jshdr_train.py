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
import argparse

from option import args
import cv2
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

import Data
import random

import datetime
oldtime = datetime.datetime.now()

#addition
from tensorflow.keras.models import Model
import tensorflow.keras.layers as KL
from tensorflow.keras import layers

import tensorflow as tf
None
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import Sequential

sess_config = tf.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)


#class ResBlock(keras.Model):
class ResBlock(object):
    #def __init__(self, filter_num, stride=1):
    def __init__(self, Channels,kSize=3,stride=1):
        #super(BasicBlock, self).__init__()
        super(ResBlock, self).__init__()
        Ch = Channels
        #model = Sequential()
        self.relu = layers.Activation('relu')

        #self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.conv1 = layers.Conv2D(filters=Ch, 
                                   padding='SAME',
                                   kernel_size=3,
                                   strides=(1, 1))
        #self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters=Ch, 
                                   padding='SAME',
                                   kernel_size=3,
                                   strides=(1, 1))
        #self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        #self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(filters=Ch, 
                                   padding='SAME',
                                   kernel_size=3,
                                   dilation_rate=2,
                                   strides=(1, 1))
        #self.conv3 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        #self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(filters=Ch, 
                                   padding='SAME',
                                   kernel_size=3,
                                   dilation_rate=2,
                                   strides=(1, 1))
        #self.conv4 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        #self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2D(filters=Ch, 
                                   padding='SAME',
                                   kernel_size=3,
                                   dilation_rate=2,
                                   strides=(1, 1))
        #self.conv5 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        #self.bn5 = layers.BatchNormalization()
        self.conv6 = layers.Conv2D(filters=Ch, 
                                   padding='SAME',
                                   kernel_size=3,
                                   dilation_rate=2,
                                   strides=(1, 1))
        #self.conv6 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        #self.bn6 = layers.BatchNormalization()

    def forward(self, x, prev_x,is_the_second,training=None):
        if is_the_second==1:
            x = x + self.relu(self.conv2(self.relu(self.conv1(x)))) + 0.1*self.relu(self.conv4(self.relu(self.conv3(x)))) + self.relu(self.conv6(self.relu(self.conv5(x))))*0.1 + prev_x
        else:
            x = x + self.relu(self.conv2(self.relu(self.conv1(x)))) + self.relu(self.conv4(self.relu(self.conv3(x))))*0.1 + self.relu(self.conv6(self.relu(self.conv5(x))))*0.1
        return x


#ResNet
class ResNet():
    def __init__(self, growRate0, nConvLayers, kSize=3):
        G0 = growRate0#64
        C  = nConvLayers
        C  = 9

        self.convs = []

        self.res1 = ResBlock(G0)
        self.convs.append(self.res1)

        self.res2 = ResBlock(G0)
        self.convs.append(self.res2)

        self.res3 = ResBlock(G0)
        self.convs.append(self.res3)

        self.res4 = ResBlock(G0)
        self.convs.append(self.res4 )

        self.res5 = ResBlock(G0)
        self.convs.append(self.res5)

        self.res6 = ResBlock(G0)
        self.convs.append(self.res6)

        self.res7 = ResBlock(G0)
        self.convs.append(self.res7)

        self.res8 = ResBlock(G0)
        self.convs.append(self.res8)

        self.res9 = ResBlock(G0)
        self.convs.append(self.res9)

        self.C = C

    def forward(self, x, feat_pre, is_the_second):
        feat_output = []
        if is_the_second==0:
            for i in range(9):
                x = self.convs[i].forward(x, [], 0)
                feat_output.append(x)
        else:
            for i in range(9):
                x = self.convs[i].forward(x, feat_pre[i], 1)
                feat_output.append(x)

        return x, feat_output


#JSHDR
class JSHDR():#object):
    def __init__(self, args):

        r = args.scale[0]
        G0 = 64
        kSize = args.RDNkSize

        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        self.encoder = layers.Conv2D(filters=G0, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1))
        #self.encoder = nn.Conv2d(6, G0, kSize, padding=(kSize-1)//2, stride=1)

        self.updater = ResNet(G0, 9)

        self.mask_estimator1 = layers.Conv2D(filters=8, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1))
        self.mask_estimator2 = layers.Conv2D(filters=2, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1))
        #self.mask_estimator1 = nn.Conv2d(G0, 8, kSize, padding=(kSize-1)//2, stride=1)
        #self.mask_estimator2 = nn.Conv2d(8, 2, kSize, padding=(kSize-1)//2, stride=1)


        self.level_estimator1 = layers.Conv2D(filters=8, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1))
        self.level_estimator2 = layers.Conv2D(filters=3, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1))
        #self.level_estimator1 = nn.Conv2d(G0, 8, kSize, padding=(kSize-1)//2, stride=1)
        #self.level_estimator2 = nn.Conv2d(8, 3, kSize, padding=(kSize-1)//2, stride=1)


        self.mask_F_w_encoder1 = layers.Conv2D(filters=16, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1))
        self.mask_F_w_encoder2 = layers.Conv2D(filters=G0, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1),
                                   kernel_initializer=tf.zeros_initializer(),
                                   bias_initializer=tf.ones_initializer()
        )
        #self.mask_F_w_encoder1 = nn.Conv2d(2, 16, kSize, padding=(kSize-1)//2, stride=1)
        #self.mask_F_w_encoder2 = nn.Conv2d(16, G0, kSize, padding=(kSize - 1) // 2, stride=1)


        self.mask_F_b_encoder1 = layers.Conv2D(filters=16, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1))
        self.mask_F_b_encoder2 = layers.Conv2D(filters=G0, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1))

        #self.mask_F_b_encoder1 = nn.Conv2d(2, 16, kSize, padding=(kSize-1)//2, stride=1)
        #self.mask_F_b_encoder2 = nn.Conv2d(16, G0, kSize, padding=(kSize - 1) // 2, stride=1)


        self.level_F_w_encoder1 = layers.Conv2D(filters=16, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1))
        self.level_F_w_encoder2 = layers.Conv2D(filters=G0, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1),
                                   kernel_initializer=tf.zeros_initializer(),
                                   bias_initializer=tf.ones_initializer()
        )

        #self.level_F_w_encoder1 = nn.Conv2d(3, 16, kSize, padding=(kSize-1)//2, stride=1)
        #self.level_F_w_encoder2 = nn.Conv2d(16, G0, kSize, padding=(kSize - 1) // 2, stride=1)


        self.level_F_b_encoder1 = layers.Conv2D(filters=16, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1))
        self.level_F_b_encoder2 = layers.Conv2D(filters=G0, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1))
        #self.level_F_b_encoder1 = nn.Conv2d(3, 16, kSize, padding=(kSize-1)//2, stride=1)
        #self.level_F_b_encoder2 = nn.Conv2d(16, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        self.decoder1 = layers.Conv2D(filters=G0, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1))
        self.decoder2 = layers.Conv2D(filters=3, 
                                   padding='SAME',
                                   kernel_size=kSize,
                                   strides=(1, 1))

        self.relu = layers.Activation('relu')
        #self.relu = nn.ReLU()

    def forward(self, x, x_prev, feat_pre, is_the_second, x_mask_prev, x_level_prev):
        x_original = x
        if is_the_second==1:
            #x = tf.concat([x,x_prev], 3)
            x = tf.keras.backend.concatenate([x,x_prev], 3)
        else:
            #x = tf.concat([x,x], 3)
            x = tf.keras.backend.concatenate([x,x], 3)

        x_F, feat_this = ResNet(64,9).forward(self.encoder(x), feat_pre, is_the_second)

        if is_the_second==1:
            x_mask = self.mask_estimator2(self.mask_estimator1(x_F)) + x_mask_prev
            x_level = self.level_estimator2(self.level_estimator1(x_F)) + x_level_prev
        else:
            x_mask = self.mask_estimator2(self.mask_estimator1(x_F))
            x_level = self.level_estimator2(self.level_estimator1(x_F))

        x_F1 = self.mask_F_b_encoder2(self.relu(self.mask_F_b_encoder1(x_mask))) + x_F \
              + self.level_F_b_encoder2(self.relu(self.level_F_b_encoder1(x_level)))
        x_F2 = self.mask_F_w_encoder2(self.relu(self.mask_F_w_encoder1(x_mask))) * x_F \
              * self.level_F_w_encoder2(self.relu(self.level_F_w_encoder1(x_level)))

        x_combine_F = x_F1 + x_F2

        myo = self.decoder1(x_combine_F)
        myo2 = self.decoder2(self.decoder1(x_combine_F))
        myo3 = myo2+x_original

        return myo3, feat_this, x_mask, x_level
        #return x_F1, feat_this, x_mask, x_level



input_x=KL.Input((64,64,3))
input_y=KL.Input((64,64,3))



kSize = 3
G0 = 64
#is_the_second = 1,first time
x_original = input_x


jshdr1 = JSHDR(args)
jshdr2 = JSHDR(args)


x1, feat_1, x_mask1, x_level1 = jshdr1.forward(input_x, [],     [], 0, [], [])
x2, feat_2, x_mask2, x_level2 = jshdr2.forward(input_x, x1, feat_1, 1, x_mask1, x_level1)



lr = input_x
hr = input_y
sr = x2
sr2 = x1
mask = x_mask2
level = x_level2

w1 = 0.0001
w2 = 0.001


def custom_loss1_L2(y_true,y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))#,keep_dims=False)
    #mse = K.mean(K.square(y_true - y_pred))
    return mse

def custom_loss1_L2_bool2num(y_true,y_pred):
    #y_true = y_true.astype(int).astype(np.float32)
    #y_pred = y_pred.astype(int).astype(np.float32)
    y_true = tf.cast(y_true,dtype=tf.float32)
    y_pred = tf.cast(y_pred,dtype=tf.float32)
    #y_true = K.cast(y_true,dtype=tf.float32)
    #y_pred = K.cast(y_pred,dtype=tf.float32)
    mse = tf.reduce_mean(tf.square(y_true - y_pred))#,keep_dims=False)
    #mse = K.mean(K.square(y_true - y_pred))
    return mse


def JointLossold(sr,sr2,hr,lr,mask,level):
    loss1_1=custom_loss1_L2(sr,hr)
    loss1_2=custom_loss1_L2(sr2,hr)
    loss1=loss1_1+loss1_2
    #per_pixel_detection_loss = custom_loss1_L2(mask, ((hr-lr)[:,:,:,0]>0))
    #per_pixel_detection_loss = custom_loss1_L2_bool2num(mask[:,:,:,0], ((hr-lr)[:,:,:,0]>0))

    #per_pixel_detection_loss = my_softmax_log_NLLLoss_sum_loss(mask, ((hr-lr)[:,:,:,0]>0))
    per_pixel_detection_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=((hr-lr)[:,:,:,0]>0),logits=mask))
    #per_pixel_detection_loss = tf.reduce_sum(per_pixel_detection_loss)
    #per_pixel_detection_loss = K.sum(per_pixel_detection_loss)
    loss2=w1*per_pixel_detection_loss
    loss3=w2*custom_loss1_L2(level, hr-lr)
    loss=loss1 + loss2 + loss3
    return loss

def JointLoss2(sr,sr2,hr,lr,mask,level):
    loss1_1=tf.reduce_mean(tf.square(sr - hr))
    loss1_2=tf.reduce_mean(tf.square(sr2- hr))
    loss1=loss1_1+loss1_2
    labels=tf.cast(((hr-lr)[:,:,:,0]>0),dtype=tf.int32)
    labels2 = tf.expand_dims(labels,-1)
    one_hot = tf.one_hot(labels,depth=2)
    sumkk = tf.reduce_sum(tf.exp(mask),3)
    sumkk = tf.expand_dims(sumkk,-1)
    softmax = tf.exp(mask)/sumkk#lr_batch1 = np.expand_dims(lr_list[i*3], axis=0)
    softmax = tf.clip_by_value(softmax,1e-10,1e100)
    logsoftmax = tf.log(softmax)
    nllloss = -tf.reduce_sum(one_hot*logsoftmax)/9#labels2.shape[0]
    per_pixel_detection_loss = nllloss#tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(((hr-lr)[:,:,:,0]>0),dtype=tf.int32),logits=mask))
    loss2=w1*per_pixel_detection_loss
    loss3=w2*tf.reduce_sum(tf.square(level - (hr-lr)))
    loss=loss1 + loss2 + loss3
    return loss,sr

loss_all,x3 = tf.keras.layers.Lambda(lambda x:JointLoss2(*x),name='loss_all')([x2,x1,input_y,input_x,x_mask2,x_level2])

model = tf.keras.models.Model([input_x,input_y],[x3])

model.get_layer('loss_all').output
model.add_loss(loss_all)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001,decay=0.0001))


def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):

    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img



def my_get_patch_9block(image, patch_size=96, scale=1, multi_scale=False):
    #ih, iw = args[0].shape[:2]
    ih, iw = 200,200#args[0].shape[:2]
    h_size, w_size = 192,192#args[0].shape[:2]

    x = resize_image(image, h_size, w_size, out_image=None, resize_mode=cv2.INTER_CUBIC)

    patch_size = 64

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

def my_get_patch9(image, patch_size=96, scale=1, multi_scale=False):
    #ih, iw = args[0].shape[:2]
    ih, iw = 200,200#args[0].shape[:2]
    h_size, w_size = 192,192#args[0].shape[:2]

    x = resize_image(image, h_size, w_size, out_image=None, resize_mode=cv2.INTER_CUBIC)

    patch_size = 64
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
    return lr_list

batch_size = 1
data = Data.data()
data2 = Data.data()
block_size = 9

my_image1 = []
my_image2 = []

for i in range(0,int(9825/batch_size)):
    #images = data.get_batch2('highlight.csv')
    #images = data.get_batch('/cache/data/highlight.csv')  # highlight.csv')
    #images = data.get_batch('/home/ma-user/modelarts/user-job-dir/code/highlight.csv',args.data_path)  # highlight.csv')
    images = data.get_batch('highlight.csv',args.data_path)  # highlight.csv')
    #print(images.shape)
    #images = Data.my_get_patch9(images[0], patch_size=96, scale=1, multi_scale=False)
    images = my_get_patch9(images[0], patch_size=96, scale=1, multi_scale=False)
    #images2 = data2.get_batch('/home/ma-user/modelarts/user-job-dir/code/nohighlight.csv',args.data_path)
    images2 = data2.get_batch('nohighlight.csv',args.data_path)
    #print(images2.shape)
    #images2 = Data.my_get_patch(images2[0], patch_size=96, scale=1, multi_scale=False)my_get_patch_9block
    images2 = my_get_patch9(images2[0], patch_size=96, scale=1, multi_scale=False)
    my_image1.extend(images)
    my_image2.extend(images2)

my_image1 = np.array(my_image1)
my_image2 = np.array(my_image2)
my_image1 = my_image1/255
my_image2 = my_image2/255
#my_image1 = my_image1[0:9]
#my_image2 = my_image2[0:9]
steps = args.steps//9 *9
my_image1 = my_image1[0:steps]
my_image2 = my_image2[0:steps]
print(my_image1.shape)
print(my_image2.shape)

#code_url = '/home/ma-user/modelarts/user-job-dir/code/'
#data_url = '/home/ma-user/modelarts/inputs/data_url_0/'
#output_url = '/home/ma-user/modelarts/outputs/train_url_0/'

checkpoint_filepath = args.output_path+'model{epoch:02d}.h5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath,monitor='val_loss',verbose=0, save_best_only=True,save_weights_only=False,mode='min', period=1)

oldtime = datetime.datetime.now()

model.fit([my_image1,my_image2],None,batch_size=9,epochs=10,callbacks=[model_checkpoint_callback])
#mox.file.copy_parallel('/home/ma-user/modelarts/outputs/train_url_0/checkp_ckpt10',)
#mox.file.copy_parallel('/home/ma-user/modelarts/outputs/train_url_0/', 'obs://schemingzero/jshdr/output8')



newtime = datetime.datetime.now()

all_time=args.steps

cost_time=(newtime-oldtime).seconds/all_time
print("sec/step : {}".format(cost_time))

print("use second")
print((newtime-oldtime).seconds)

sess.close()




