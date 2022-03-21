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
#python news.py --data_dir=data --batch_size=1 --mode=cmc
#python news.py --mode=test --image1=data/labeled/val/0046_00.jpg --image2=data/labeled/val/0049_07.jpg
from npu_bridge.npu_init import *
import tensorflow as tf
import numpy as np
import cv2
import big_dataset_label as cuhk03_dataset_label2
import random
import cmc
from triplet_loss import batch_hard_triplet_loss
import time

from importlib import import_module
from tensorflow.contrib import slim
from nets import NET_CHOICES
from heads import HEAD_CHOICES

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib.pyplot as plt  
from PIL import Image 

print(tf.__version__)
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '30', 'batch size for training')
tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')
tf.flags.DEFINE_string('logs_dir', 'logs_paper_1x1_a_1x7/', 'path to logs directory')
tf.flags.DEFINE_string('data_dir', 'data_eye/', 'path to dataset')
tf.flags.DEFINE_float('learning_rate', '0.01', '')
tf.flags.DEFINE_string('mode', 'train', 'Mode train, val, test')
tf.flags.DEFINE_string('image1', '', 'First image path to compare')
tf.flags.DEFINE_string('image2', '', 'Second image path to compare')

tf.flags.DEFINE_float('global_rate', '1.0', 'global rate')
tf.flags.DEFINE_float('local_rate', '1.0', 'local rate')
tf.flags.DEFINE_float('softmax_rate', '1.0', 'softmax rate')

tf.flags.DEFINE_integer('ID_num', '10', 'id number')
tf.flags.DEFINE_integer('IMG_PER_ID', '3', 'img per id')



tf.flags.DEFINE_integer('embedding_dim', '128', 'Dimensionality of the embedding space.')
#tf.flags.DEFINE_string('initial_checkpoint', 'resnet_v1_50.ckpt', 'Path to the checkpoint file of the pretrained network.')

tf.flags.DEFINE_string('initial_checkpoint2', 'resnet_v1_101.ckpt', 'Path to the checkpoint file of the pretrained network.')



IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224



def preprocess(images, is_train):
    def train():    
        split = tf.split(images, [1, 1,1])
        shape = [1 for _ in xrange(split[0].get_shape()[1])]
        for i in xrange(len(split)):
            split[i] = tf.reshape(split[i], [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3])
            split[i] = tf.split(split[i], shape)
            for j in xrange(len(split[i])):
                #split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT , IMAGE_WIDTH , 3])
                split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT + 8, IMAGE_WIDTH + 3, 3])
                split[i][j] = tf.random_crop(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i][j] = tf.image.random_flip_left_right(split[i][j])
                split[i][j] = tf.image.random_brightness(split[i][j], max_delta=32. / 255.)
                split[i][j] = tf.image.random_saturation(split[i][j], lower=0.5, upper=1.5)
                split[i][j] = tf.image.random_hue(split[i][j], max_delta=0.2)
                split[i][j] = tf.image.random_contrast(split[i][j], lower=0.5, upper=1.5)
                split[i][j] = tf.image.per_image_standardization(split[i][j])
        return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[2], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
    def val():
        split = tf.split(images, [1, 1,1])
        shape = [1 for _ in xrange(split[0].get_shape()[1])]
        for i in xrange(len(split)):
            split[i] = tf.reshape(split[i], [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
            split[i] = tf.image.resize_images(split[i], [IMAGE_HEIGHT, IMAGE_WIDTH])
            split[i] = tf.split(split[i], shape)
            for j in xrange(len(split[i])):
                split[i][j] = tf.reshape(split[i][j], [IMAGE_HEIGHT, IMAGE_WIDTH, 3])
                split[i][j] = tf.image.per_image_standardization(split[i][j])
        return [tf.reshape(tf.concat(split[0], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3]),
            tf.reshape(tf.concat(split[1], axis=0), [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3])]
    return tf.cond(is_train, train, val)




def compute_euclidean_distance(x, y):
    """
    Computes the euclidean distance between two tensorflow variables
    """

    #    x    #Tensor("network/l2_normalize:0", shape=(10, 512), dtype=float32)
    d = tf.square(tf.subtract(x, y))     # shape=(10, 512)
    d = tf.sqrt(tf.reduce_sum(d,1)) # What about the axis ???
    return d

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        #pos_cos_similarity = tf.reduce_sum(tf.multiply(anchor,positive),1) #  1 : similarity     0 : not  similarity
        #pos_cos_similarity = 1 - pos_cos_similarity  # 0: similarity   1 :not  similarity
        #neg_cos_similarity = tf.reduce_sum(tf.multiply(anchor,negative),1)
        #neg_cos_similarity =1 - neg_cos_similarity
        #basic_loss = tf.add(tf.subtract(pos_cos_similarity,neg_cos_similarity), alpha)
        a = tf.square(tf.subtract(anchor, positive))#shape=(128, 2048)
        print (a,'   aaaaaaaaaaa    aaaaaaaaaaaaa ')
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)# shape=(128,)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        '''
        top_64 = 64
        value = []
        size_a = tf.size(pos_dist)
        max_index = tf.nn.top_k(pos_dist, size_a)[1]
        index = max_index[:top_64]
        for i in range(top_64):
            j = index[i]
            value.append([pos_dist[j]])
        pos_tensor = tf.convert_to_tensor(value, dtype=tf.float32)
        pos_tensor_top64 = tf.reshape(pos_tensor,[top_64,])
        
        
        #http://blog.csdn.net/noirblack/article/details/78088993
        value = []
        size_a = tf.size(neg_dist)
        min_index = tf.nn.top_k(-neg_dist, size_a)[1]
        index = min_index[:top_64]
        for i in range(top_64):
            j = index[i]
            value.append([neg_dist[j]])
        neg_tensor = tf.convert_to_tensor(value, dtype=tf.float32)
        neg_tensor_top64 = tf.reshape(neg_tensor,[top_64,])
        '''


        #basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        
      
    return loss,tf.reduce_mean(pos_dist),tf.reduce_mean(neg_dist)




def fully_connected_class(features,feature_dim,num_classes):
    # Higher-Order Relationships
    with slim.variable_scope.variable_scope("ball", reuse=None):
            weights = slim.model_variable(
                "mean_vectors", (feature_dim, int(num_classes)),
                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer=None)
            scale = slim.model_variable(
                "scale", (), tf.float32,
                initializer=tf.constant_initializer(0., tf.float32),
                regularizer=slim.l2_regularizer(1e-1))
            
            scale = tf.nn.softplus(scale)

    # Mean vectors in colums, normalize axis 0.
    weights_normed = tf.nn.l2_normalize(weights, dim=0)
    logits = scale * tf.matmul(features, weights_normed)
    return logits


def fully_connected_class2(features,feature_dim,num_classes):
    # Higher-Order Relationships
    with slim.variable_scope.variable_scope("ball2", reuse=None):
            weights = slim.model_variable(
                "mean_vectors2", (feature_dim, int(num_classes)),
                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer=None)
            scale = slim.model_variable(
                "scale2", (), tf.float32,
                initializer=tf.constant_initializer(0., tf.float32),
                regularizer=slim.l2_regularizer(1e-1))
            
            scale = tf.nn.softplus(scale)

    # Mean vectors in colums, normalize axis 0.
    weights_normed = tf.nn.l2_normalize(weights, dim=0)
    logits = scale * tf.matmul(features, weights_normed)
    
    return logits





def global_pooling(images1,images2,weight_decay ):
    with tf.variable_scope('network_global_pool'):
        # Tied Convolution    
        global_pool = 7
    
        #conv1_branch1 = tf.layers.conv2d(images1, 512, [1, 1], reuse=None, name='conv1_branch1')        
        feat1_avg_pool1 = tf.nn.avg_pool(images1, ksize=[1, global_pool, global_pool, 1], strides=[1, 1, 1, 1], padding='VALID')
        #feat1_avg_pool1 = tf.nn.avg_pool(feat1_prod1, ksize=[1, global_pool, global_pool, 1], strides=[1, global_pool, global_pool, 1], padding='SAME')
        reshape_branch1 = tf.reshape(feat1_avg_pool1, [FLAGS.batch_size, -1])
        
        
        
        #conv2_branch1 = tf.layers.conv2d(images2, 2048, [1, 1], reuse=True, name='conv1_branch1')        
        feat2_avg_pool1 = tf.nn.avg_pool(images2, ksize=[1, global_pool, global_pool, 1], strides=[1, 1, 1, 1], padding='VALID')
        #feat2_avg_pool1 = tf.nn.avg_pool(feat2_prod1, ksize=[1, global_pool, global_pool, 1], strides=[1, global_pool, global_pool, 1], padding='SAME')
        reshape2_branch1 = tf.reshape(feat2_avg_pool1, [FLAGS.batch_size, -1])
        
        
        '''
        #conv3_branch1 = tf.layers.conv2d(images3, 2048, [1, 1], reuse=True, name='conv1_branch1')
        feat3_avg_pool1 = tf.nn.avg_pool(images3, ksize=[1, global_pool, global_pool, 1], strides=[1, 1, 1, 1], padding='VALID')
        reshape3_branch1 = tf.reshape(feat3_avg_pool1, [FLAGS.batch_size, -1])
        
        '''
        
        concat1_L2 = tf.nn.l2_normalize(reshape_branch1,dim=1)
        
        concat2_L2 = tf.nn.l2_normalize(reshape2_branch1,dim=1)
        
        #concat3_L2 = tf.nn.l2_normalize(reshape3_branch1,dim=1) 

        #return concat1_L2,concat2_L2,concat3_L2
        return concat1_L2,concat2_L2                                                                                                                                                                                                       

    
    

    
    
def global_pooling2(images1,weight_decay ):
    with tf.variable_scope('network_global_pool2'):
        # Tied Convolution    
        global_pool = 7
    
        #conv1_branch1 = tf.layers.conv2d(images1, 512, [1, 1], reuse=None, name='conv1_branch1')        
        feat1_avg_pool1 = tf.nn.avg_pool(images1, ksize=[1, global_pool, global_pool, 1], strides=[1, 1, 1, 1], padding='VALID')
        #feat1_avg_pool1 = tf.nn.avg_pool(feat1_prod1, ksize=[1, global_pool, global_pool, 1], strides=[1, global_pool, global_pool, 1], padding='SAME')
        reshape_branch1 = tf.reshape(feat1_avg_pool1, [FLAGS.batch_size, -1])
        
        
        '''
        #conv2_branch1 = tf.layers.conv2d(images2, 2048, [1, 1], reuse=True, name='conv1_branch1')        
        feat2_avg_pool1 = tf.nn.avg_pool(images2, ksize=[1, global_pool, global_pool, 1], strides=[1, 1, 1, 1], padding='VALID')
        #feat2_avg_pool1 = tf.nn.avg_pool(feat2_prod1, ksize=[1, global_pool, global_pool, 1], strides=[1, global_pool, global_pool, 1], padding='SAME')
        reshape2_branch1 = tf.reshape(feat2_avg_pool1, [FLAGS.batch_size, -1])
  
        #conv3_branch1 = tf.layers.conv2d(images3, 2048, [1, 1], reuse=True, name='conv1_branch1')
        feat3_avg_pool1 = tf.nn.avg_pool(images3, ksize=[1, global_pool, global_pool, 1], strides=[1, 1, 1, 1], padding='VALID')
        reshape3_branch1 = tf.reshape(feat3_avg_pool1, [FLAGS.batch_size, -1])
        
        '''
        
        concat1_L2 = tf.nn.l2_normalize(reshape_branch1,dim=1)
        
        #concat2_L2 = tf.nn.l2_normalize(reshape2_branch1,dim=1)
        
        #concat3_L2 = tf.nn.l2_normalize(reshape3_branch1,dim=1) 

        #return concat1_L2,concat2_L2,concat3_L2
        return concat1_L2                                   
    
    
    
    
    
    
    
def local_pooling(images1,images2,weight_decay ):
    with tf.variable_scope('network_local_pool'):
        # Tied Convolution    
        global_pool = 1  #h
        local_pool = 7   #w
    
        #conv1_branch1 = tf.layers.conv2d(images1, 2048, [1, 1],  reuse=False, name='conv1_branch1')        
        feat1_avg_pool1 = tf.nn.avg_pool(images1, ksize=[1, global_pool, local_pool, 1], strides=[1, 1, 1, 1], padding='VALID')

        conv1_1 = tf.layers.conv2d(feat1_avg_pool1, 128, [1, 1],padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), reuse=None, name='conv1x1')     

        reshape_branch1 = tf.reshape(conv1_1, [FLAGS.batch_size, -1])
        
             
        
        #conv2_branch1 = tf.layers.conv2d(images2, 2048, [1, 1], reuse=True, name='conv1_branch1')        
        feat2_avg_pool1 = tf.nn.avg_pool(images2, ksize=[1, global_pool, local_pool, 1], strides=[1, 1, 1, 1], padding='VALID')
        conv2_1 = tf.layers.conv2d(feat2_avg_pool1, 128, [1, 1],padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), reuse=None, name='conv1x1_')

        reshape2_branch1 = tf.reshape(conv2_1, [FLAGS.batch_size, -1])
              
        
        concat1_L2 = tf.nn.l2_normalize(reshape_branch1,dim=1)

        concat2_L2 = tf.nn.l2_normalize(reshape2_branch1,dim=1)
              
       
        normal_1 = tf.reshape(concat1_L2, [FLAGS.batch_size, -1,128])
        normal_2 = tf.reshape(concat2_L2, [FLAGS.batch_size, -1,128])
       
        return normal_1,normal_2





def tf_compute_local_distance(anchor_feature , positive_feature , negative_feature):
    list_ = []
    for i in range(7):
        for j in range(7):
            anchor_feature_seg = anchor_feature[:,i]    #  anchor_feature>>(batch,7,128)     anchor_feature[:,i]>>(batch,1,128) 
        
            positive_feature_seg = positive_feature[:,j]
    
            pos_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(anchor_feature_seg, positive_feature_seg)), 1))# shape=(128,)
            
            #temp_array[i,j] = pos_dist 
            
            list_.append(pos_dist)

    trans_list = tf.transpose(list_)  # list_ (7x7,batch)    >>        trans_list (batch,7x7)

    re_list = tf.reshape(trans_list,[FLAGS.batch_size,7,7])  #   re_list (batch,7,7)
    local_p = tf.div( tf.exp(re_list)- 1 , tf.exp(re_list)+ 1 )
    
    #local pos
    m=7
    n=7
    dist = [[0 for _ in range(n)] for _ in range(m)]     
    for a in range(m):
        for b in range(n):
            if (a == 0) and (b == 0):
                dist[a][b] = local_p[:,a, b]
            elif (a == 0) and (b > 0):
                dist[a][b] = dist[a][b - 1] + local_p[:,a, b]
            elif (a > 0) and (b == 0):
                dist[a][b] = dist[a - 1][b] + local_p[:,a, b]
            else:
                dist[a][b] = tf.minimum(dist[a - 1][b], dist[a][b - 1]) + local_p[:,a, b]
    dist = dist[-1][-1]    
    


    
    list_2 = []
    for i in range(7):
        for j in range(7):
            anchor_feature_seg = anchor_feature[:,i]    #  anchor_feature>>(batch,7,128)     anchor_feature[:,i]>>(batch,1,128) 
        
            negative_feature_seg = negative_feature[:,j]
    
            negative_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(anchor_feature_seg, negative_feature_seg)), 1))# shape=(128,)
            
            #temp_array[i,j] = pos_dist 
            
            list_2.append(negative_dist)
  
    trans_list2 = tf.transpose(list_2)  # list_ (7x7,batch)    >>        trans_list (batch,7x7)
    re_list2 = tf.reshape(trans_list2,[FLAGS.batch_size,7,7])  #   re_list (batch,7,7)
    local_n = tf.div( tf.exp(re_list2)- 1 , tf.exp(re_list2)+ 1 )
    
    # local neg
    m=7
    n=7
    dist2 = [[0 for _ in range(n)] for _ in range(m)]   
    for a in range(m):
        for b in range(n):
            if (a == 0) and (b == 0):
                dist2[a][b] = local_n[:,a, b]
            elif (a == 0) and (b > 0):
                dist2[a][b] = dist2[a][b - 1] + local_n[:,a, b]
            elif (a > 0) and (b == 0):
                dist2[a][b] = dist2[a - 1][b] + local_n[:,a, b]
            else:
                dist2[a][b] = tf.minimum(dist2[a - 1][b], dist2[a][b - 1]) + local_n[:,a, b]
    dist2 = dist2[-1][-1]
    
    return dist,dist2

def local_triplet(pos_dist,neg_dist,alpha):
    with tf.variable_scope('local_triplet'):
         
         print('pos_dist',pos_dist)
         print('neg_dist',neg_dist)
         
         basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
         loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
         print('basic_loss',  basic_loss)
         print('loss',    loss)
         return loss,tf.reduce_mean(pos_dist),tf.reduce_mean(neg_dist)










def triplet_hard_loss(y_pred,id_num,img_per_id):
    with tf.variable_scope('hard_triplet'):

        SN = img_per_id  #img per id
        PN =id_num   #id num
        feat_num = SN*PN # images num
        
        #y_pred = tf.nn.l2_normalize(y_pred,dim=1) 
    
        feat1 = tf.tile(tf.expand_dims(y_pred,0),[feat_num,1,1])
        feat2 = tf.tile(tf.expand_dims(y_pred,1),[1,feat_num,1])
        
        delta = tf.subtract(feat1,feat2)
        dis_mat = tf.reduce_sum(tf.square(delta), 2)+ 1e-8

        dis_mat = tf.sqrt(dis_mat)
     
        #dis_mat = tf.reduce_sum(tf.square(tf.subtract(feat1, feat2)), 2)
        #dis_mat = tf.sqrt(dis_mat)
        

    
        positive = dis_mat[0:SN,0:SN]
        negetive = dis_mat[0:SN,SN:]
        
        for i in range(1,PN):
            positive = tf.concat([positive,dis_mat[i*SN:(i+1)*SN,i*SN:(i+1)*SN]],axis = 0)
            if i != PN-1:
                negs = tf.concat([dis_mat[i*SN:(i+1)*SN,0:i*SN],dis_mat[i*SN:(i+1)*SN, (i+1)*SN:]],axis = 1)
            else:
                negs = tf.concat(dis_mat[i*SN:(i+1)*SN, 0:i*SN],axis = 0)
            negetive = tf.concat([negetive,negs],axis = 0)
  
        p=positive
        n=negetive
        positive = tf.reduce_max(positive,1)
        negetive = tf.reduce_min(negetive,axis=1) #acc
        
        #negetive = tf.reduce_mean(negetive,1)
        #negetive = tf.reduce_max(negetive,axis=1) #false

        a1 = 0.3
        
        #basic_loss = tf.add(tf.subtract(positive,negetive), a1)
        #loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        #loss = tf.reduce_mean(tf.maximum(0.0,positive-negetive+a1))
        
        
        # effective loss
        effective = tf.maximum(0.0,positive-negetive+a1)
        num_active = tf.reduce_sum(tf.cast(tf.greater(effective, 1e-5), tf.float32))
        loss_mean = tf.reduce_mean(effective)
        #effective_loss_mean = (loss_mean * FLAGS.batch_size) / num_active
        
        effective_loss_mean = tf.divide( (loss_mean * FLAGS.batch_size) ,num_active)
        
       
        return effective_loss_mean ,dis_mat 
        

        
        
        
        
        
        
        
def local_triplet_hard_loss(y_pred,id_num,img_per_id):


    SN = img_per_id  #img per id
    PN =id_num   #id num
    feat_num = SN*PN # images num     
    list_ = [] 

    for i in range(7):
        for j in range(7):
            
            feat1 = tf.tile(tf.expand_dims(y_pred,0),[feat_num,1,1,1])
            feat2 = tf.tile(tf.expand_dims(y_pred,1),[1,feat_num,1,1])
            
            #delta = tf.subtract(feat1,feat2)
            #dis_mat = tf.reduce_sum(tf.square(delta), 2)+ 1e-8
            #dis_mat = tf.sqrt(dis_mat)
            
            anchor_feature_seg = feat1[:,:,i]    #  anchor_feature>>(batch,7,128)     anchor_feature[:,i]>>(batch,1,128)      
            positive_feature_seg = feat2[:,:,j] 
            pos_dist = tf.sqrt( 1e-8+tf.reduce_sum(tf.square(tf.subtract(anchor_feature_seg, positive_feature_seg)), 2))# shape=(128,)
            list_.append(pos_dist)
    
    
    trans_list = tf.transpose(list_)  # list_ (7x7,batch)    >>        trans_list (batch,7x7)
    
    re_list = tf.reshape(trans_list,[feat_num*feat_num,7,7])  #   re_list (batch,7,7)
    
    local_p = tf.div( tf.exp(re_list)- 1 , tf.exp(re_list)+ 1 )
    
    #local pos
    m=7
    n=7
    dist = [[0 for _ in range(n)] for _ in range(m)]     
    for a in range(m):
        for b in range(n):
            if (a == 0) and (b == 0):
                dist[a][b] = local_p[:,a, b]
            elif (a == 0) and (b > 0):
                dist[a][b] = dist[a][b - 1] + local_p[:,a, b]
            elif (a > 0) and (b == 0):
                dist[a][b] = dist[a - 1][b] + local_p[:,a, b]
            else:
                dist[a][b] = tf.minimum(dist[a - 1][b], dist[a][b - 1]) + local_p[:,a, b]
    dist = dist[-1][-1]    
    
    dis_mat = tf.reshape(dist,[feat_num,feat_num]) 
    
    #pick 
    positive = dis_mat[0:SN,0:SN]
    negetive = dis_mat[0:SN,SN:]
    for i in range(1,PN):
        positive = tf.concat([positive,dis_mat[i*SN:(i+1)*SN,i*SN:(i+1)*SN]],axis = 0)
        if i != PN-1:
            negs = tf.concat([dis_mat[i*SN:(i+1)*SN,0:i*SN],dis_mat[i*SN:(i+1)*SN, (i+1)*SN:]],axis = 1)
        else:
            negs = tf.concat(dis_mat[i*SN:(i+1)*SN, 0:i*SN],axis = 0)
        negetive = tf.concat([negetive,negs],axis = 0)

    
    positive = tf.reduce_max(positive,1)
    negetive = tf.reduce_min(negetive,axis=1) 

    a1 = 0.3
    #loss = tf.reduce_mean(tf.maximum(0.0,positive-negetive+a1))
    
    # effective loss
    effective = tf.maximum(0.0,positive-negetive+a1)
    num_active = tf.reduce_sum(tf.cast(tf.greater(effective, 1e-5), tf.float32))
    loss_mean = tf.reduce_mean(effective)
    #effective_loss_mean = (loss_mean * FLAGS.batch_size) / num_active
    effective_loss_mean = tf.cond(num_active < 0.001, lambda: loss_mean , lambda: tf.divide( (loss_mean * FLAGS.batch_size) ,num_active))
    
    
    return effective_loss_mean ,tf.reduce_mean(positive) ,tf.reduce_mean(negetive)     
        

        
def multual_loss(matrix1,matrix2):
    
    delta1 = tf.square( tf.subtract(matrix1,matrix2) )
    delta2 = tf.square( tf.subtract(matrix2,matrix1) )
    
    dis1 = tf.reduce_sum(delta1)
    dis2 = tf.reduce_sum(delta2)
    
    loss = tf.divide( (dis1+dis2) ,(FLAGS.batch_size * FLAGS.batch_size) )
    
    return loss

def kl_loss_compute(logits1, logits2):
    """ KL loss
    """
    pred1 = tf.nn.softmax(logits1)
    pred2 = tf.nn.softmax(logits2)
    loss = tf.reduce_mean(tf.reduce_sum(pred2 * tf.log(1e-8 + pred2 / (pred1 + 1e-8)), 1))

    return loss

def main(argv=None):

    if FLAGS.mode == 'test':
        FLAGS.batch_size = 1
    
    if FLAGS.mode == 'cmc':
        FLAGS.batch_size = 1

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    #images = tf.placeholder(tf.float32, [2, FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
    images = tf.placeholder(tf.float32, [3, FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
    
    images_total = tf.placeholder(tf.float32, [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images_total')
    
    labels = tf.placeholder(tf.float32, [FLAGS.batch_size], name='labels')
    #labels_neg = tf.placeholder(tf.float32, [FLAGS.batch_size, 743], name='labels')
    
    #total
    #labels = tf.placeholder(tf.float32, [FLAGS.batch_size, 847], name='labels')
    #labels_neg = tf.placeholder(tf.float32, [FLAGS.batch_size, 847], name='labels')
    
    #eye
    #labels = tf.placeholder(tf.float32, [FLAGS.batch_size, 104], name='labels')
    #labels_neg = tf.placeholder(tf.float32, [FLAGS.batch_size, 104], name='labels')

    
    
    is_train = tf.placeholder(tf.bool, name='is_train')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    weight_decay = 0.0005
    tarin_num_id = 0
    val_num_id = 0

    if FLAGS.mode == 'train':
        tarin_num_id = cuhk03_dataset_label2.get_num_id(FLAGS.data_dir, 'train')
        print(tarin_num_id, '               11111111111111111111               1111111111111111')
    elif FLAGS.mode == 'val':
        val_num_id = cuhk03_dataset_label2.get_num_id(FLAGS.data_dir, 'val')

    
    # Create the model and an embedding head.
    model = import_module('nets.' + 'resnet_v1_50')
    head = import_module('heads.' + 'fc1024')
    
    
    # Feed the image through the model. The returned `body_prefix` will be used
    # further down to load the pre-trained weights for all variables with this
    # prefix.
    endpoints, body_prefix = model.endpoints(images_total, is_training=True)

    with tf.name_scope('head'):
        endpoints = head.head(endpoints, FLAGS.embedding_dim, is_training=True)
    
    
    '''
    print endpoints['model_output'] # (bt,2048)
    print endpoints['global_pool'] # (bt,2048)
    print endpoints['resnet_v1_50/block4']# (bt,7,7,2048)
    '''

    # Create the model and an embedding head.
    model2 = import_module('nets.' + 'resnet_v1_101')
    endpoints2, body_prefix2 = model2.endpoints(images_total, is_training=True)
       
    train_mode = tf.placeholder(tf.bool)

    print('Build network')
    
    feat = endpoints['resnet_v1_50/block4']# (bt,7,7,2048)
    
    feat2 = endpoints2['resnet_v1_101/block4']# (bt,7,7,2048)

    #feat = tf.convert_to_tensor(feat, dtype=tf.float32)
    # global
    feature,feature2 = global_pooling(feat,feat2,weight_decay)
    loss_triplet ,PP,NN = batch_hard_triplet_loss(labels,feature,0.3)
    
    
    _,dis_matrix1 = triplet_hard_loss(feature,FLAGS.ID_num,FLAGS.IMG_PER_ID)
    _,dis_matrix2 = triplet_hard_loss(feature2,FLAGS.ID_num,FLAGS.IMG_PER_ID)
    mul_loss = multual_loss(dis_matrix1,dis_matrix2)

    
    
    local_anchor_feature, local_anchor_feature2 = local_pooling(feat,feat2,weight_decay)
    local_loss_triplet ,local_pos_loss, local_neg_loss = local_triplet_hard_loss(local_anchor_feature,FLAGS.ID_num,FLAGS.IMG_PER_ID)
    
    
    
    loss_triplet2 ,PP2,NN2 = batch_hard_triplet_loss(labels,feature2,0.3)
    local_loss_triplet2 ,local_pos_loss2, local_neg_loss2 = local_triplet_hard_loss(local_anchor_feature2,FLAGS.ID_num,FLAGS.IMG_PER_ID)
    

    
    s1 = fully_connected_class(feature,feature_dim=2048,num_classes=1000)#tarin_num_id
    cross_entropy_var = slim.losses.sparse_softmax_cross_entropy(s1, tf.cast(labels, tf.int64))
    loss_softmax = cross_entropy_var
    #loss_softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_softmax, logits=s1))
    inference = tf.nn.softmax(s1)
    
    s2 = fully_connected_class2(feature2,feature_dim=2048,num_classes=1000)
    cross_entropy_var2 = slim.losses.sparse_softmax_cross_entropy(s2, tf.cast(labels, tf.int64))
    loss_softmax2 = cross_entropy_var2
    
    #loss_softmax2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels_softmax, logits=s2))
    inference2 = tf.nn.softmax(s2)
    
    multual_softmax1 = kl_loss_compute(s1, s2)
    multual_softmax2 = kl_loss_compute(s2, s1)
    
    
    
    
    P1= tf.reduce_mean(PP)
    P2= tf.reduce_mean(PP2)
    N1= tf.reduce_mean(NN)
    N2= tf.reduce_mean(NN2)
    
    LP1= tf.reduce_mean(local_pos_loss)
    LN1= tf.reduce_mean(local_neg_loss)
    
    
    
    '''
    
    # global
    feature2 = global_pooling(feat2,weight_decay)
    #loss_triplet,PP,NN = triplet_hard_loss(feature,FLAGS.ID_num,FLAGS.IMG_PER_ID)
    loss_triplet2 ,PP2,NN2 = batch_hard_triplet_loss(labels,feature2,0.3)

    
    #local
    local_anchor_feature2 = local_pooling(feat2,weight_decay)
    local_loss_triplet2 ,local_pos_loss2, local_neg_loss2 = local_triplet_hard_loss(local_anchor_feature2,FLAGS.ID_num,FLAGS.IMG_PER_ID)
    '''
    
    

    loss = local_loss_triplet*FLAGS.local_rate + loss_triplet*FLAGS.global_rate + mul_loss + loss_softmax + multual_softmax1
   
    #DD = compute_euclidean_distance(anchor_feature,positive_feature)
    loss2 = local_loss_triplet2*FLAGS.local_rate + loss_triplet2*FLAGS.global_rate + mul_loss + loss_softmax2 + multual_softmax2
    

    
    
    if FLAGS.mode == 'val' or FLAGS.mode == 'cmc' or FLAGS.mode == 'test':
       loss ,pos_loss, neg_loss = triplet_loss(anchor_feature, positive_feature, negative_feature, 0.3)
       print(' ERROR                 ERROR ')
       None
    

    
    
    
    
    # These are collected here before we add the optimizer, because depending
    # on the optimizer, it might add extra slots, which are also global
    # variables, with the exact same prefix.
    model_variables = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, body_prefix)
    
    model_variables2 = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, body_prefix2)
    
      
    
    # Update_ops are used to update batchnorm stats.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        

    
        optimizer = npu_tf_optimizer(tf.train.MomentumOptimizer(learning_rate, momentum=0.9))
        train = optimizer.minimize(loss, global_step=global_step)
        
        optimizer2 = npu_tf_optimizer(tf.train.MomentumOptimizer(learning_rate, momentum=0.9))
        train2 = optimizer2.minimize(loss2, global_step=global_step)
    

    tf.summary.scalar("total_loss 1", loss)
    tf.summary.scalar("total_loss 2", loss2)
    tf.summary.scalar("learning_rate", learning_rate)

    regularization_var = tf.reduce_sum(tf.losses.get_regularization_loss())
    tf.summary.scalar("weight_loss", regularization_var)
    


    lr = FLAGS.learning_rate

    #config=tf.ConfigProto(log_device_placement=True)
    #config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)) 
    # GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    # custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    # custom_op.name = "NpuOptimizer"
    # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")

    with tf.Session(config=npu_config_proto(config_proto=config)) as sess:
        
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("TensorBoard_1x1_a_1x7/", graph = sess.graph)

        #sess.run(tf.global_variables_initializer())
        #saver = tf.train.Saver()
        
        #checkpoint_saver = tf.train.Saver(max_to_keep=0)
        checkpoint_saver = tf.train.Saver()


        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore model')
            print(ckpt.model_checkpoint_path)
            #saver.restore(sess, ckpt.model_checkpoint_path)
            checkpoint_saver.restore(sess, ckpt.model_checkpoint_path)
                    
        #for first , training load imagenet
        else:
            sess.run(tf.global_variables_initializer())
            #saver = tf.train.Saver(model_variables)
            #print FLAGS.initial_checkpoint
            #saver.restore(sess, FLAGS.initial_checkpoint)
            
            
            saver2 = tf.train.Saver(model_variables2)
            print(FLAGS.initial_checkpoint2)
            saver2.restore(sess, FLAGS.initial_checkpoint2)
   
            
            
        if FLAGS.mode == 'train':
            step = sess.run(global_step)
            for i in range(step, FLAGS.max_steps + 1):

                batch_images, batch_labels, batch_images_total = cuhk03_dataset_label2.read_data(FLAGS.data_dir, 'train', tarin_num_id,
                    IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size,FLAGS.ID_num,FLAGS.IMG_PER_ID)
                
                #feed_dict = {learning_rate: lr,  is_train: True , labels: batch_labels, droup_is_training: False, train_mode: True, images_total: batch_images_total} #no label   images: batch_images,
              
                feed_dict = {learning_rate: lr,  is_train: True , train_mode: True, images_total: batch_images_total, labels: batch_labels}

                                              
                start = time.time()
                                
                _,_,train_loss,train_loss2 = sess.run([train,train2,loss,loss2 ], feed_dict=feed_dict) 
                    
                print('Step: %d, Learning rate: %f, Train loss: %f , Train loss2: %f' % (i, lr, train_loss,train_loss2))
                
                gtoloss,gp,gn = sess.run([loss_triplet,P1,N1], feed_dict=feed_dict)   
                print('global hard: ',gtoloss)
                print('global P1: ',gp)
                print('global N1: ',gn)
                             
                toloss,p,n = sess.run([local_loss_triplet,LP1,LN1], feed_dict=feed_dict)   
                print('local hard: ',toloss)
                print('local P: ',p)
                print('local N: ',n)
                                
                mul,p2,n2 = sess.run([mul_loss,loss_triplet2,local_loss_triplet2], feed_dict=feed_dict)   
                print('mul loss: ',mul)
                print('loss_triplet2: ',p2)
                print('local_loss_triplet2: ',n2)                               
                
                end = time.time()
                elapsed = end - start
                print("Time taken: ", elapsed, "seconds.")
                                
               
                #lr = FLAGS.learning_rate / ((2) ** (i/160000)) * 0.1
                lr = FLAGS.learning_rate * ((0.0001 * i + 1) ** -0.75)
                if i % 100 == 0:
               
                    checkpoint_saver.save(sess,FLAGS.logs_dir + 'model.ckpt', i)
                
                if i % 20 == 0:
                    result = sess.run(merged, feed_dict=feed_dict)
                    writer.add_summary(result, i)
                
                
        

        elif FLAGS.mode == 'val':
            total = 0.
            for _ in xrange(10):
                batch_images, batch_labels = cuhk03_dataset_label2.read_data(FLAGS.data_dir, 'val', val_num_id,
                    IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size)
                feed_dict = {images: batch_images, labels: batch_labels, is_train: False}
                prediction = sess.run(inference, feed_dict=feed_dict)
                prediction = np.argmax(prediction, axis=1)
                label = np.argmax(batch_labels, axis=1)

                for i in xrange(len(prediction)):
                    if prediction[i] == label[i]:
                        total += 1
            print('Accuracy: %f' % (total / (FLAGS.batch_size * 10)))

            '''
            for i in xrange(len(prediction)):
                print('Prediction: %s, Label: %s' % (prediction[i] == 0, labels[i] == 0))
                image1 = cv2.cvtColor(batch_images[0][i], cv2.COLOR_RGB2BGR)
                image2 = cv2.cvtColor(batch_images[1][i], cv2.COLOR_RGB2BGR)
                image = np.concatenate((image1, image2), axis=1)
                cv2.imshow('image', image)
                key = cv2.waitKey(0)
                if key == 1048603:  # ESC key
                    break
            '''

        
        elif FLAGS.mode == 'cmc':    
          do_times = 1
          cmc_sum=np.zeros((100, 100), dtype='f')
          for times in xrange(do_times):  
              path = 'data' 
              set = 'val'
              
              cmc_array=np.ones((100, 100), dtype='f')
              
              batch_images = []
              batch_labels = []
              index_gallery_array=np.ones((1, 100), dtype='f')
              gallery_bool = True
              probe_bool = True
              for j in xrange(100):
                      id_probe = j
                      for i in xrange(100):
                              batch_images = []
                              batch_labels = []
                              filepath = ''
                              
                              #filepath_gallery = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, i, index_gallery)
                              #filepath_probe = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, id_probe, index_probe)                          
                              
                              if gallery_bool == True:
                                    while True:
                                          index_gallery = int(random.random() * 10)
                                          index_gallery_array[0,i] = index_gallery
  
                                          filepath_gallery = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, i, index_gallery)
                                          if not os.path.exists(filepath_gallery):
                                              continue
                                          break
                              if i ==99:
                                  gallery_bool = False
                              if gallery_bool == False:
                                          index_gallery = index_gallery_array[0,i]
                                          filepath_gallery = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, i, index_gallery)
                              
                              
                              
                              if probe_bool == True:
                                    while True:
                                          index_probe = int(random.random() * 10)
                                          filepath_probe = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, id_probe, index_probe)
                                          if not os.path.exists(filepath_probe):
                                              continue
                                          if index_gallery_array[0,id_probe] == index_probe:
                                              continue
                                          probe_bool = False
                                          break
                              if i ==99:
                                  probe_bool = True
                              
                              
                              '''
                              while True:
                                    index_probe = int(random.random() * 10)
                                    filepath_probe = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, id_probe, index_probe)
                                    if not os.path.exists(filepath_gallery):
                                        continue
                                    if index_gallery_array[1,id_probe] == index_probe:
                                        continue
                                    break
                              '''
                              
                              #filepath_gallery = 'data/labeled/val/0000_01.jpg'
                              #filepath_probe   = 'data/labeled/val/0000_02.jpg'
                                                                          
                              image1 = cv2.imread(filepath_gallery)
                              image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
                              image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                              image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
                              
                              image2 = cv2.imread(filepath_probe)
                              image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
                              image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                              image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
                              
                              test_images = np.array([image1, image2, image2])
                              
                              #print (filepath_gallery)
                              #print (filepath_probe)
                              #print ('1111111111111111111111')
          
                              if i == j:
                                  batch_labels = [1., 0.]
                              if i != j:    
                                  batch_labels = [0., 1.]
                              batch_labels = np.array(batch_labels)
                              print('test  img :',test_images.shape)
                              
                              feed_dict = {images: test_images, is_train: False}
                              prediction = sess.run(DD, feed_dict=feed_dict)
                              #print (prediction, prediction[0][1])
                              
                              print (filepath_gallery,filepath_probe)
                              
                              #print(bool(not np.argmax(prediction[0])))
                              print (prediction)
                              
                              cmc_array[j,i] = prediction
                              
                              #print(i,j)
                             
                              
                              #prediction = sess.run(inference, feed_dict=feed_dict)
                              #prediction = np.argmax(prediction, axis=1)
                              #label = np.argmax(batch_labels, axis=1)
                              
  
              
              cmc_score = cmc.cmc(cmc_array)
              cmc_sum = cmc_score + cmc_sum
              print(cmc_score)
          cmc_sum = cmc_sum/do_times
          print(cmc_sum)
          print('final cmc') 
        
        
        
        elif FLAGS.mode == 'test':
            image1 = cv2.imread(FLAGS.image1)
            image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            image2 = cv2.imread(FLAGS.image2)
            image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            test_images = np.array([image1, image2,image2])

            feed_dict = {images: test_images, is_train: False, droup_is_training: False}
            #prediction, prediction2 = sess.run([DD,DD2], feed_dict=feed_dict)
            prediction = sess.run([inference], feed_dict=feed_dict)
            prediction = np.array(prediction)
            print(prediction.shape)
            print( np.argmax(prediction[0])+1)
            
           
        
            #print(bool(not np.argmax(prediction[0])))

if __name__ == '__main__':
    tf.app.run()