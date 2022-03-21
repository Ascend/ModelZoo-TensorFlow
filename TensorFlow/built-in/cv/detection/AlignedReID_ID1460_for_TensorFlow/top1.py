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

#import cuhk03_dataset_label2
import big_dataset_label as cuhk03_dataset_label2

import random
import cmc

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
tf.flags.DEFINE_integer('batch_size', '80', 'batch size for training')
tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')
tf.flags.DEFINE_string('logs_dir', 'logs_paper_1x1_a_1x7/', 'path to logs directory')
tf.flags.DEFINE_string('data_dir', 'data/', 'path to dataset')
tf.flags.DEFINE_float('learning_rate', '0.01', '')
tf.flags.DEFINE_string('mode', 'top1', 'Mode train, val, test')
tf.flags.DEFINE_string('image1', '', 'First image path to compare')
tf.flags.DEFINE_string('image2', '', 'Second image path to compare')

tf.flags.DEFINE_float('global_rate', '1.0', 'global rate')
tf.flags.DEFINE_float('local_rate', '1.0', 'local rate')
tf.flags.DEFINE_float('softmax_rate', '1.0', 'softmax rate')

tf.flags.DEFINE_integer('ID_num', '20', 'id number')
tf.flags.DEFINE_integer('IMG_PER_ID', '4', 'img per id')

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
                #split[i][j] = tf.image.per_image_standardization(split[i][j])
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





def fully_connected_class(anchor_feature , positive_feature , negative_feature):
    # Higher-Order Relationships
    reshape = tf.reshape(anchor_feature, [FLAGS.batch_size, -1])
    fc3 = tf.layers.dense(reshape, 743,reuse=None, name='fc3')
    
    
    reshape_pos = tf.reshape(positive_feature, [FLAGS.batch_size, -1])
    fc3_pos = tf.layers.dense(reshape_pos, 743,reuse=True, name='fc3')
    
    reshape_neg = tf.reshape(negative_feature, [FLAGS.batch_size, -1])
    fc3_neg = tf.layers.dense(reshape_neg, 743,reuse=True, name='fc3')
    
    return fc3, fc3_pos, fc3_neg





def global_pooling(images1,weight_decay ):
    with tf.variable_scope('network_global_pool', reuse = True):
        # Tied Convolution    
        global_pool = 7
        feat1_avg_pool1 = tf.nn.avg_pool(images1, ksize=[1, global_pool, global_pool, 1], strides=[1, 1, 1, 1], padding='VALID')
        #feat1_avg_pool1 = tf.nn.avg_pool(feat1_prod1, ksize=[1, global_pool, global_pool, 1], strides=[1, global_pool, global_pool, 1], padding='SAME')
        reshape_branch1 = tf.reshape(feat1_avg_pool1, [FLAGS.batch_size, -1])   
        concat1_L2 = tf.nn.l2_normalize(reshape_branch1,dim=1)
        return concat1_L2                                                                                                                                                                                                        

def local_pooling(images1,weight_decay ):
    with tf.variable_scope('network_local_pool'):
        # Tied Convolution    
        global_pool = 1
        local_pool = 7
    
        #conv1_branch1 = tf.layers.conv2d(images1, 2048, [1, 1],  reuse=False, name='conv1_branch1')        
        feat1_avg_pool1 = tf.nn.avg_pool(images1, ksize=[1, global_pool, local_pool, 1], strides=[1, 1, 1, 1], padding='VALID')
        conv1_1 = tf.layers.conv2d(feat1_avg_pool1, 128, [7, 1],padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), reuse=None, name='conv1x1')     
        reshape_branch1 = tf.reshape(conv1_1, [FLAGS.batch_size, -1])      
        concat1_L2 = tf.nn.l2_normalize(reshape_branch1,dim=1)        
        normal_1 = tf.reshape(concat1_L2, [FLAGS.batch_size, -1,128])   
        return normal_1



def main(argv=None):
    if FLAGS.mode == 'test':
        FLAGS.batch_size = 1
    
    if FLAGS.mode == 'cmc':
        FLAGS.batch_size = 1
        
    if FLAGS.mode == 'top1':
        FLAGS.batch_size = 100

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
   
    images = tf.placeholder(tf.float32, [3, FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
    
    images_total = tf.placeholder(tf.float32, [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images_total')
    

    images_one = tf.placeholder(tf.float32, [1, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images_one')


    
    
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
  
    images1, images2,images3 = preprocess(images, is_train)
    img_combine = tf.concat([images1, images2,images3], 0)
    
    train_mode = tf.placeholder(tf.bool)
       
    # Create the model and an embedding head.
    model = import_module('nets.' + 'resnet_v1_50')
    head = import_module('heads.' + 'fc1024')
        
    # Feed the image through the model. The returned `body_prefix` will be used
    # further down to load the pre-trained weights for all variables with this
    # prefix.
    endpoints, body_prefix = model.endpoints(images_total, is_training=False)

    feat = endpoints['resnet_v1_50/block4']# (bt,7,7,2048)
    
    print('Build network')

    # global
    anchor_feature = global_pooling(feat,weight_decay)
   
    lr = FLAGS.learning_rate

    #config=tf.ConfigProto(log_device_placement=True)
    #config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)) 
    # GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.55
    
    with tf.Session(config=config) as sess:
        checkpoint_saver = tf.train.Saver(max_to_keep=0)

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

        
        if FLAGS.mode == 'train':
            step = sess.run(global_step)
            for i in xrange(step, FLAGS.max_steps + 1):

                batch_images, batch_labels, batch_images_total = cuhk03_dataset_label2.read_data(FLAGS.data_dir, 'train', tarin_num_id,
                    IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size,FLAGS.ID_num,FLAGS.IMG_PER_ID)
                
       
                feed_dict = {learning_rate: lr,  is_train: True , train_mode: True, images_total: batch_images_total}

                
                
                
                _,train_loss = sess.run([train,loss], feed_dict=feed_dict) 
                    
                print('Step: %d, Learning rate: %f, Train loss: %f ' % (i, lr, train_loss))
                
                
                
                
                h,p,l = sess.run([NN,PP,loss], feed_dict=feed_dict)   
                print 'n:',h
                print 'p:',p
                print 'hard loss',l
                

                
                
                
                lr = FLAGS.learning_rate * ((0.0001 * i + 1) ** -0.75)
                if i % 100 == 0:
                    saver.save(sess, FLAGS.logs_dir + 'model.ckpt', i)
       



                
        elif FLAGS.mode == 'top1':
            path = 'data_eye' 
            set = 'val'
            cmc_sum=np.zeros((100, 100), dtype='f')

            cmc_total = []
            do_times = 20

            for times in xrange(do_times):  
                query_feature = []
                test_feature = []

                for i in range(100):
                    while True:
                          index_gallery = int(random.random() * 10)
                          index_temp = index_gallery
                          filepath_gallery = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, i, index_gallery)
                          if not os.path.exists(filepath_gallery):
                             continue
                          break
                    image1 = cv2.imread(filepath_gallery)
                    image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                    query_feature.append(image1)
    
                    while True:
                          index_gallery = int(random.random() * 10)
                          if index_temp == index_gallery:
                             continue
      
                          filepath_gallery = '%s/labeled/%s/%04d_%02d.jpg' % (path, set, i, index_gallery)
                          if not os.path.exists(filepath_gallery):
                             continue
                          break
                    image1 = cv2.imread(filepath_gallery)
                    image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                    test_feature.append(image1)
                    #print filepath_gallery,'\n'
                query_feature = np.array(query_feature)
                test_feature = np.array(test_feature)
          
                feed_dict = {images_total: query_feature, is_train: False}
                q_feat = sess.run(anchor_feature, feed_dict=feed_dict)
                
                feed_dict = {images_total: test_feature, is_train: False}
                test_feat = sess.run(anchor_feature, feed_dict=feed_dict)
    
                cmc_array = []
                tf_q_feat = tf.constant(q_feat)
                tf_test_feat = tf.constant(test_feat)
  
                h = tf.placeholder(tf.int32)
                pick = tf_q_feat[h]
                tf_q_feat = tf.reshape(pick,[1,2048])
                feat1 = tf.tile(tf_q_feat,[100,1])
                f = tf.square(tf.subtract(feat1 , tf_test_feat))
                d = tf.sqrt(tf.reduce_sum(f,1)) # What about the axis ???
                print d       ,'f\n'     
                for t in range(100):
                    
                    feed_dict = {h: t}
                    D = sess.run(d,feed_dict=feed_dict)
                    cmc_array.append(D)
                cmc_array = np.array(cmc_array)
                cmc_score = cmc.cmc(cmc_array)
                cmc_sum = cmc_score + cmc_sum
                cmc_total.append(cmc_score)
                #top1=single_query(q_feat,test_feat,labels,labels,test_num=10)
                print cmc_score
            cmc_sum = cmc_sum/do_times
            print(cmc_sum)
            print('final cmc') 
            print ('\n')
            print cmc_total
        
        
      

if __name__ == '__main__':
    tf.app.run()