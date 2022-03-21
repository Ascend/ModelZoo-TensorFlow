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

# from src.dataflow.market import MarketTriplet
# import os
# from npu_bridge.npu_init import *
from npu_bridge.npu_init import *
import numpy as np
import skimage.transform
import tensorflow as tf
# from config import *
# from model import *
slim = tf.contrib.slim
from data import *
from loss import *
from tensorflow.contrib.slim import nets
import moxing as mox
import os


import argparse
arg = argparse.ArgumentParser()
# expriment setting 
arg.add_argument('-e','--expriment_name',type=str,default='test')
arg.add_argument('-modelname','--modelname',type=str,default='softmaxcircleloss')
arg.add_argument('-epoch','--epoch_size',type=int,default=60)
# arg.add_argument('-g','--gpu',type=str,default='0,1,2,3,4,5,6,7')

# hyper
arg.add_argument('-w','--warmup',type=int,default=5)
arg.add_argument('-b','--batch_size',type=int,default=32)
arg.add_argument('-lr','--init_lr',type=float,default=0.05)
arg.add_argument('-lamda','--lamda',type=float,default=1)
arg.add_argument('-drop','--droprate',type=float,default=0.5)

# for huawei npu training
arg.add_argument("--train_url", type=str, default="./output")
arg.add_argument("--data_url", type=str, default="./dataset")
# arg.add_argument("--ckpt_url", type=str, default="./ckpt")
arg.add_argument("--result_dir", type=str, default="/cache/results")
# arg.add_argument("--ckpt_dir", type=str, default="/cache/ckpt")
arg.add_argument("--data_dir", type=str, default="/cache/")
arg.add_argument("--ckptpname", type=str, default="resnet_v1_50.ckpt")

args = arg.parse_args()

im_size = [256, 128]
warmup = args.warmup
m=0.25
gamma=32
lamda = args.lamda
num_classes=751
epoch_size = args.epoch_size
batch_size = args.batch_size
init_lr = args.init_lr
num_bottleneck = 512

def makedirifnotexist(path):
    if not os.path.exists(path):
        os.makedirs(path)


# backup code
# log_dir
# from tool import copy_code
# copy_code(src="/home/liulizhao/projects/liuyixin/projects",tar=log_dir+"code/")

# 在ModelArts容器创建数据存放目录
data_dir = args.data_dir
makedirifnotexist(data_dir)
datasetpath = os.path.join(data_dir,'Market-1501-v15.09.15','bounding_box_train')
# OBS数据拷贝到ModelArts容器内
mox.file.copy_parallel(args.data_url, data_dir)   
# 预训练模型
# ckpt_dir = args.ckpt_dir
# os.makedirs(ckpt_dir)
# mox.file.copy_parallel(args.ckpt_url, ckpt_dir)  
ckptpath = os.path.join(data_dir,'ckpt',args.ckptpname)

# 在ModelArts容器创建训练输出目录
result_dir = args.result_dir
makedirifnotexist(result_dir)
# 该实验的目录
expriment_dir = os.path.join(result_dir,args.expriment_name)
makedirifnotexist(expriment_dir)
# 相关log
log_dir = os.path.join(expriment_dir,'logs')
makedirifnotexist(log_dir)
# 相关ckpt
ckpt_output_dir = os.path.join(expriment_dir,'ckpt')
makedirifnotexist(ckpt_output_dir)

x = tf.placeholder(tf.float32, [None, im_size[0],im_size[1],3],name='inputs')
y_ = tf.placeholder(tf.int32, [None],name='labels')
lr_backbone = tf.placeholder(tf.float32, name='lr_backbone')
lr_logit = tf.placeholder(tf.float32, name='lr_logit')
is_training = tf.placeholder(tf.bool, name='is_training')
keep_rate = tf.placeholder(tf.float32, name='keep_rate')


# 读入数据
imnum, dataset,le = tf_dataset(batch_size=batch_size,epoch_size=None,path=datasetpath)
print(f'total class num :{len(le.classes_)}')
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
    net, endpoints = nets.resnet_v1.resnet_v1_50(x, num_classes=None,
                                                    is_training=is_training,global_pool=True)
import math
with tf.variable_scope('Logits'):
    # batch_norm_params = {"is_training": is_training, "decay": 0.9, "updates_collections": None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(5e-4)):
        resnetfeat = tf.squeeze(net, axis=[1, 2])
        fc1 = slim.fully_connected(resnetfeat, num_outputs=num_bottleneck,
                                        activation_fn=None,weights_initializer =tf.random_normal_initializer(mean=0,stddev=math.sqrt(2/num_bottleneck)))
        bn1 = slim.batch_norm(fc1, activation_fn=None,is_training=is_training)
        drop1 = slim.dropout(bn1, keep_prob=keep_rate)
        norm_feat = tf.math.l2_normalize(drop1,axis=1,name='features',epsilon=1e-12)
        logits = slim.fully_connected(drop1, num_outputs=num_classes,
                                        activation_fn=None,weights_initializer=tf.random_normal_initializer(mean=0,stddev=0.001))

checkpoint_exclude_scopes = 'Logits'
exclusions = None
if checkpoint_exclude_scopes:
    exclusions = [
        scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
variables_to_restore = []
add_var = []
outof = []

for var in slim.get_model_variables():
    excluded = False
    for exclusion in exclusions:
        if var.op.name.startswith(exclusion):
            excluded = True
    if not excluded:
        variables_to_restore.append(var)
    else:
        add_var.append(var)

softmaxloss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y_, logits=logits)
# softmaxloss = tf.reduce_mean(softmaxloss)
# labels = tf.one_hot(y_,num_classes)
# soft_logit = tf.nn.softmax(logits)
# softmaxloss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
# soft_logits = tf.clip_by_value(soft_logit, 0.000001, 0.999999)
# cross_entropy = -tf.reduce_sum(labels * tf.log(soft_logits), axis=1)
# softmaxloss = tf.reduce_mean(cross_entropy)
circle_loss_op,sp,sn,recall,precision,f1_score = PairWiseCircleLoss(norm_feat,y_)
circle_loss_term = lamda * circle_loss_op/batch_size
# loss = softmaxloss + circle_loss_term
loss = softmaxloss + circle_loss_term
# loss = softmaxloss 

classes = tf.argmax(logits, axis=1, name='classes')
accuracy_softmax = tf.reduce_mean(tf.cast(
    tf.equal(tf.cast(classes, dtype=tf.int32), y_), dtype=tf.float32))
optimizer1 = tf.train.MomentumOptimizer(learning_rate=lr_backbone,momentum=0.9,use_nesterov=True)
optimizer2 = tf.train.MomentumOptimizer(learning_rate=lr_logit,momentum=0.9,use_nesterov=True)

# bn var
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step1 = optimizer1.minimize(loss,var_list=variables_to_restore)
    train_step2 = optimizer2.minimize(loss,var_list=add_var)

init = tf.global_variables_initializer()

saver_restore = tf.train.Saver(var_list=variables_to_restore)
saver = tf.train.Saver(tf.global_variables())

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True      #程序按需申请内存

# tensorboard记录下来
# tf.summary.histogram('sn',sn)
# tf.summary.histogram('sp',sp)
# # recall,precision,f1_score
# tf.summary.scalar('pairwise_recall',recall)
# tf.summary.scalar('pairwise_precision',precision)
# tf.summary.scalar('pairwise_f1_score',f1_score)
# tf.summary.scalar('softmaxloss',softmaxloss)
# # tf.summary.scalar('reg_loss',reg_term)
# tf.summary.scalar('circleloss',circle_loss_term)
# tf.summary.scalar('totalloss',loss)
# tf.summary.scalar('accuracy_softmax',accuracy_softmax)
# tf.summary.scalar('lr_backbone',lr_backbone)
# tf.summary.scalar('lr_logit',lr_logit)
# merged = tf.summary.merge_all()

def adjust_LR(step,epoch):
    lr_l = init_lr * 0.1**((epoch)//40)
    lr_b = 0.1 * lr_l
    if epoch < warmup:
        factor = min(1.0, 0.1 + 0.9 / (imnum//batch_size*warmup))
        lr_l,lr_b = factor*lr_l,factor*lr_b
    # print(f'学习率: lr_l:{lr_l}, lr_b:{lr_b}!')
    return lr_l,lr_b

# 开始训练
with tf.Session(config=npu_config_proto(config_proto=config)) as sess:
    
    # writter = tf.summary.FileWriter(log_dir, sess.graph)
    # train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)

    # 执行初始化操作
    sess.run(init)
    
    # Load the pretrained checkpoint file xxx.ckpt
    saver_restore.restore(sess, ckptpath)

    epoch = 0
    sample_counter = 0
    step_counter = 0
    # for i in range(step_size):
    while epoch < epoch_size:
        step_counter+=1
        import time
        t0=time.time()
        images, groundtruth_lists = sess.run(one_element)
        # print(images[0])
        # print(groundtruth_lists)
        # print(groundtruth_lists)
        batch_now = images.shape[0]
        sample_counter+=batch_now
        epoch = sample_counter // imnum
        if batch_now < batch_size: # skip the last batch
            print('skip last')
            continue
        lr_l,lr_b = adjust_LR(step_counter,epoch)

        train_dict = {x: images, 
                        y_: groundtruth_lists,
                        is_training: True,
                        lr_logit:lr_l,
                        lr_backbone:lr_b,
                        keep_rate:0.5}
        # summary,loss_, acc_,_,_ = sess.run([merged,loss, accuracy_softmax,train_step1,train_step2], feed_dict=train_dict)
        sotfmaxloss_,circle_loss_,loss_, acc_, _, _ = sess.run([ softmaxloss,circle_loss_term,loss, accuracy_softmax, train_step1, train_step2],
                                              feed_dict=train_dict)
        # sotfmaxloss_,loss_, acc_, _, _ = sess.run([ softmaxloss,loss, accuracy_softmax, train_step1, train_step2],
                                            #   feed_dict=train_dict)
        # pred,gt = sess.run([classes,y_],feed_dict=train_dict)
        # print(f'gt is {gt} and model pred is {pred}!')
        # train_writer.add_summary(summary, step_counter)
        # train_writer.add_summary(summary, step_counter)
        
        t1 = time.time()
        
        train_text = 'Epoch:{}, Step: {}, train_Loss: {:.4f}, train_Accuracy: {:.4f}, Timing:{:.4f}s'.format(
            epoch,step_counter+1, loss_, acc_,t1-t0)
        print(train_text)
        print(f'softmaxloss:{sotfmaxloss_}   circleloss:{circle_loss_}')

        if (step_counter+1) % 500 == 0:
            saver.save(sess, ckpt_output_dir+'resnet50_circleloss.ckpt', global_step=step_counter+1)
            tf.io.write_graph(sess.graph, ckpt_output_dir, 'graph.pbtxt', as_text=True)
            print('save mode to {}'.format(ckpt_output_dir))


# 训练结束后，将ModelArts容器内的训练输出拷贝到OBS
mox.file.copy_parallel(result_dir, args.train_url) 

# train_writer.close()




