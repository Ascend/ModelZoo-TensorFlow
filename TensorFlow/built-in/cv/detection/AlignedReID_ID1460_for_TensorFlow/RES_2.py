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

print tf.__version__
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '36', 'batch size for training')
tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')
tf.flags.DEFINE_string('logs_dir', 'logs_paper_1x1_a_1x7/', 'path to logs directory')
tf.flags.DEFINE_string('data_dir', 'data/', 'path to dataset')
tf.flags.DEFINE_float('learning_rate', '0.01', '')
tf.flags.DEFINE_string('mode', 'train', 'Mode train, val, test')
tf.flags.DEFINE_string('image1', '', 'First image path to compare')
tf.flags.DEFINE_string('image2', '', 'Second image path to compare')

tf.flags.DEFINE_float('global_rate', '1.0', 'global rate')
tf.flags.DEFINE_float('local_rate', '1.0', 'local rate')
tf.flags.DEFINE_float('softmax_rate', '1.0', 'softmax rate')

tf.flags.DEFINE_integer('ID_num', '6', 'id number')
tf.flags.DEFINE_integer('IMG_PER_ID', '6', 'img per id')



tf.flags.DEFINE_integer('embedding_dim', '128', 'Dimensionality of the embedding space.')
tf.flags.DEFINE_string('initial_checkpoint', 'resnet_v1_50.ckpt', 'Path to the checkpoint file of the pretrained network.')

tf.flags.DEFINE_string('initial_checkpoint2', 'resnet_v1_101.ckpt', 'Path to the checkpoint file of the pretrained network.')



IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224




def fully_connected_class(features,feature_dim,num_classes):
    
    # Higher-Order Relationships
    with slim.variable_scope.variable_scope("ball", reuse=None):
            weights = slim.model_variable(
                "mean_vectors", (feature_dim, feature_dim*2),
                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer=None)
            scale = slim.model_variable(
                "scale", (), tf.float32,
                initializer=tf.constant_initializer(0., tf.float32),
                regularizer=slim.l2_regularizer(1e-1))
            
            scale = tf.nn.relu(scale)#scale = tf.nn.softplus(scale)

    # Mean vectors in colums, normalize axis 0.
    weights_normed = tf.nn.l2_normalize(weights, dim=0)
    logits = scale * tf.matmul(features, weights_normed)
    
    
    
    #2 layer  no activate
    with slim.variable_scope.variable_scope("b2", reuse=None):
            weights2 = slim.model_variable(
                "mean_vectors2", (feature_dim*2, int(num_classes)),
                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer=None)
            scale2 = slim.model_variable(
                "scale2", (), tf.float32,
                initializer=tf.constant_initializer(0., tf.float32),
                regularizer=slim.l2_regularizer(1e-1))
            
    # Mean vectors in colums, normalize axis 0.
    weights_normed2 = tf.nn.l2_normalize(weights2, dim=0)
    logits =  tf.matmul(logits, weights_normed2)       
    
    
    
    
    return logits


def fully_connected_class2(features,feature_dim,num_classes):
    # Higher-Order Relationships
    with slim.variable_scope.variable_scope("ball2", reuse=None):
            weights = slim.model_variable(
                "mean_vectors22", (feature_dim, feature_dim*2),
                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer=None)
            scale = slim.model_variable(
                "scale22", (), tf.float32,
                initializer=tf.constant_initializer(0., tf.float32),
                regularizer=slim.l2_regularizer(1e-1))
            
            scale = tf.nn.relu(scale)#scale = tf.nn.softplus(scale)

    # Mean vectors in colums, normalize axis 0.
    weights_normed = tf.nn.l2_normalize(weights, dim=0)
    logits = scale * tf.matmul(features, weights_normed)
    
    
        
    #2 layer  no activate
    with slim.variable_scope.variable_scope("ball22", reuse=None):
            weights2 = slim.model_variable(
                "mean_vectors22", (feature_dim*2, int(num_classes)),
                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer=None)
            scale2 = slim.model_variable(
                "scale22", (), tf.float32,
                initializer=tf.constant_initializer(0., tf.float32),
                regularizer=slim.l2_regularizer(1e-1))
            
    # Mean vectors in colums, normalize axis 0.
    weights_normed2 = tf.nn.l2_normalize(weights2, dim=0)
    logits =  tf.matmul(logits, weights_normed2) 
    
    
    
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






def local_triplet(pos_dist,neg_dist,alpha):
    with tf.variable_scope('local_triplet'):
         
         print 'pos_dist',pos_dist
         print 'neg_dist',neg_dist
         
         basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
         loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
         print 'basic_loss',  basic_loss  
         print 'loss',    loss  
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
        
        basic_loss = tf.add(tf.subtract(positive,negetive), a1)
        effective_loss_mean = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        #loss = tf.reduce_mean(tf.maximum(0.0,positive-negetive+a1)) #Equation up
        
        
        '''
        # effective loss
        effective = tf.maximum(0.0,positive-negetive+a1)
        num_active = tf.reduce_sum(tf.cast(tf.greater(effective, 1e-5), tf.float32))
        loss_mean = tf.reduce_mean(effective)
        #effective_loss_mean = (loss_mean * FLAGS.batch_size) / num_active
        
        effective_loss_mean = tf.divide( (loss_mean * FLAGS.batch_size) ,num_active)
        '''
        
       
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
       print ' ERROR                 ERROR '
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
        

    
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        train = optimizer.minimize(loss, global_step=global_step)
        
        optimizer2 = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
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

    with tf.Session(config=config) as sess:
        
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("TensorBoard_1x1_a_1x7/", graph = sess.graph)

        #sess.run(tf.global_variables_initializer())
        #saver = tf.train.Saver()
        
        #checkpoint_saver = tf.train.Saver(max_to_keep=0)
        checkpoint_saver = tf.train.Saver()


        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore model')
            print ckpt.model_checkpoint_path
            #saver.restore(sess, ckpt.model_checkpoint_path)
            checkpoint_saver.restore(sess, ckpt.model_checkpoint_path)
                    
        #for first , training load imagenet
        else:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(model_variables)
            print FLAGS.initial_checkpoint
            saver.restore(sess, FLAGS.initial_checkpoint)
            
            
            saver2 = tf.train.Saver(model_variables2)
            print FLAGS.initial_checkpoint2
            saver2.restore(sess, FLAGS.initial_checkpoint2)
   
            
            
        if FLAGS.mode == 'train':
            step = sess.run(global_step)
            for i in xrange(step, FLAGS.max_steps + 1):

                batch_images, batch_labels, batch_images_total = cuhk03_dataset_label2.read_data(FLAGS.data_dir, 'train', tarin_num_id,
                    IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size,FLAGS.ID_num,FLAGS.IMG_PER_ID)
                
                #feed_dict = {learning_rate: lr,  is_train: True , labels: batch_labels, droup_is_training: False, train_mode: True, images_total: batch_images_total} #no label   images: batch_images,
              
                feed_dict = {learning_rate: lr,  is_train: True , train_mode: True, images_total: batch_images_total, labels: batch_labels}

                                              
                start = time.time()
                                
                _,_,train_loss,train_loss2 = sess.run([train,train2,loss,loss2 ], feed_dict=feed_dict) 
                
                
                    
                print('Step: %d, Learning rate: %f, Train loss: %f , Train loss2: %f' % (i, lr, train_loss,train_loss2))
                
                gtoloss,gp,gn = sess.run([loss_triplet,P1,N1], feed_dict=feed_dict)   
                print 'global hard: ',gtoloss
                print 'global P1: ',gp
                print 'global N1: ',gn
                             
                toloss,p,n = sess.run([local_loss_triplet,LP1,LN1], feed_dict=feed_dict)   
                print 'local hard: ',toloss
                print 'local P: ',p
                print 'local N: ',n
                                
                mul,p2,n2 = sess.run([mul_loss,loss_triplet2,local_loss_triplet2], feed_dict=feed_dict)   
                print 'mul loss: ',mul
                print 'loss_triplet2: ',p2
                print 'local_loss_triplet2: ',n2                               
                
                end = time.time()
                elapsed = end - start
                print "Time taken: ", elapsed, "seconds."
                                
               
                #lr = FLAGS.learning_rate / ((2) ** (i/160000)) * 0.1
                lr = FLAGS.learning_rate * ((0.0001 * i + 1) ** -0.75)
                if i % 100 == 0:
               
                    checkpoint_saver.save(sess,FLAGS.logs_dir + 'model.ckpt', i)
                
                if i % 20 == 0:
                    result = sess.run(merged, feed_dict=feed_dict)
                    writer.add_summary(result, i)
                
                

if __name__ == '__main__':
    tf.app.run()