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
# from npu_bridge.estimator.npu import npu_convert_dropout
import numpy as np
import math
import random
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import glob
from tqdm import tqdm
import pickle
import scipy.ndimage.interpolation as inter
from scipy.signal import medfilt 
from scipy.spatial.distance import cdist

# from keras.models import Model
# from keras.layers import *
# from keras.layers.core import *
# from keras.layers.convolutional import *
# from keras import backend as K

# import keras
import tensorflow as tf
npu_keras_sess = set_keras_session_npu_config()
random.seed(1234)

class Config():
    def __init__(self):
        self.frame_l = 32 # the length of frames
        self.joint_n = 15 # the number of joints
        self.joint_d = 2 # the dimension of joints
        self.clc_num = 21 # the number of class
        self.feat_d = 105
        self.filters = 64
C = Config()

# Temple resizing function
def zoom(p,target_l=64,joints_num=25,joints_dim=3):
    l = p.shape[0]
    p_new = np.empty([target_l,joints_num,joints_dim]) 
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:,m,n] = medfilt(p[:,m,n],3)
            p_new[:,m,n] = inter.zoom(p[:,m,n],target_l/l)[:target_l]         
    return p_new



# Calculate JCD feature
def norm_scale(x):
    return (x-np.mean(x))/np.mean(x)
  
def get_CG(p,C):
    M = []
    iu = np.triu_indices(C.joint_n,1,C.joint_n)
    for f in range(C.frame_l): 
        d_m = cdist(p[f],p[f],'euclidean')       
        d_m = d_m[iu] 
        M.append(d_m)
    M = np.stack(M) 
    M = norm_scale(M)
    return M
  
  
# Genrate dataset  
def data_generator(T,C,le):
    X_0 = []
    X_1 = []
    Y = []
    for i in tqdm(range(len(T['pose']))): 
        p = np.copy(T['pose'][i])
        p = zoom(p,target_l=C.frame_l,joints_num=C.joint_n,joints_dim=C.joint_d)

        label = np.zeros(C.clc_num)
        label[le.transform(T['label'])[i]-1] = 1   

        M = get_CG(p,C)

        X_0.append(M)
        X_1.append(p)
        Y.append(label)

    X_0 = np.stack(X_0)  
    X_1 = np.stack(X_1) 
    Y = np.stack(Y)
    return X_0,X_1,Y

def poses_diff(x):
    H, W = x.get_shape()[1],x.get_shape()[2]
    x = tf.subtract(x[:,1:,...],x[:,:-1,...])
    x = tf.image.resize(x,size=[H,W]) 
    return x

def pose_motion(P,frame_l):
    P_diff_slow = tf.keras.layers.Lambda(lambda x: poses_diff(x))(P)
    P_diff_slow = tf.keras.layers.Reshape((frame_l,30))(P_diff_slow)
    P_fast = tf.keras.layers.Lambda(lambda x: x[:,::2,...])(P)
    P_diff_fast = tf.keras.layers.Lambda(lambda x: poses_diff(x))(P_fast)
    P_diff_fast = tf.keras.layers.Reshape((int(frame_l/2),30))(P_diff_fast)
    return P_diff_slow,P_diff_fast
    
def c1D(x,filters,kernel):
    x = tf.keras.layers.Conv1D(filters, kernel_size=kernel,padding='same',use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x

def block(x,filters):
    x = c1D(x,filters,3)
    x = c1D(x,filters,3)
    return x
    
def d1D(x,filters):
    x = tf.keras.layers.Dense(filters,use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x

def build_FM(frame_l=32,joint_n=22,joint_d=2,feat_d=231,filters=16):   
    M = tf.keras.Input(shape=(frame_l,feat_d))
    P = tf.keras.Input(shape=(frame_l,joint_n,joint_d))
    
    diff_slow,diff_fast = pose_motion(P,frame_l)
    
    x = c1D(M,filters*2,1)
    x = tf.keras.layers.SpatialDropout1D(0.1)(x)
    x = c1D(x,filters,3)
    x = tf.keras.layers.SpatialDropout1D(0.1)(x)
    x = c1D(x,filters,1)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.SpatialDropout1D(0.1)(x)

    x_d_slow = c1D(diff_slow,filters*2,1)
    x_d_slow = tf.keras.layers.SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow,filters,3)
    x_d_slow = tf.keras.layers.SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow,filters,1)
    x_d_slow = tf.keras.layers.MaxPool1D(2)(x_d_slow)
    x_d_slow = tf.keras.layers.SpatialDropout1D(0.1)(x_d_slow)
        
    x_d_fast = c1D(diff_fast,filters*2,1)
    x_d_fast = tf.keras.layers.SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast,filters,3) 
    x_d_fast = tf.keras.layers.SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast,filters,1) 
    x_d_fast = tf.keras.layers.SpatialDropout1D(0.1)(x_d_fast)
   
    x = tf.keras.layers.concatenate([x,x_d_slow,x_d_fast])
    x = block(x,filters*2)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.SpatialDropout1D(0.1)(x)
    
    x = block(x,filters*4)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.SpatialDropout1D(0.1)(x)

    x = block(x,filters*8)
    x = tf.keras.layers.SpatialDropout1D(0.1)(x)
    
    return tf.keras.Model(inputs=[M,P],outputs=x)


def build_DD_Net(C):
    M = tf.keras.Input(name='M', shape=(C.frame_l,C.feat_d))  
    P = tf.keras.Input(name='P', shape=(C.frame_l,C.joint_n,C.joint_d)) 
    
    FM = build_FM(C.frame_l,C.joint_n,C.joint_d,C.feat_d,C.filters)
    
    x = FM([M,P])

    x = tf.keras.layers.GlobalMaxPool1D()(x)
    
    x = d1D(x,128)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = d1D(x,128)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(C.clc_num, activation='softmax')(x)
    
    ######################Self-supervised part
    model = tf.keras.Model(inputs=[M,P],outputs=x)
    return model

DD_Net = build_DD_Net(C)
DD_Net.summary()

Train = pickle.load(open("data/JHMDB/GT_train_1.pkl", "rb"))
Test = pickle.load(open("data/JHMDB/GT_test_1.pkl", "rb"))

# Train = pickle.load(open("data/SHREC/train.pickle", "rb"))
# Test = pickle.load(open("data/SHREC/test.pickle", "rb"))
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(Train['label'])

X_0,X_1,Y = data_generator(Train,C,le)
X_test_0,X_test_1,Y_test = data_generator(Test,C,le)

lr = 1e-3
DD_Net.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adam(lr),metrics=['accuracy'])
lrScheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, cooldown=5, min_lr=5e-6)
history = DD_Net.fit([X_0,X_1],Y,
                    batch_size=len(Y),
                    epochs=600,
                    verbose=True,
                    shuffle=True,
                    callbacks=[lrScheduler],
                    validation_data=([X_test_0,X_test_1],Y_test)      
                    )
lr = 1e-4
DD_Net.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adam(lr),metrics=['accuracy'])
lrScheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, cooldown=5, min_lr=5e-6)
history = DD_Net.fit([X_0,X_1],Y,
                    batch_size=len(Y),
                    epochs=600,
                    verbose=True,
                    shuffle=True,
                    callbacks=[lrScheduler],
                    validation_data=([X_test_0,X_test_1],Y_test)      
                    )
close_session(npu_keras_sess)
