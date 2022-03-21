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

import time
import numpy as np
 
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
 

from src.dataflow.market import MarketTriplet
import numpy as np
import skimage.transform
import tensorflow as tf
from config import *
from model import *
from data import *
from loss import *

from tool import copy_code
copy_code(src="/home/liulizhao/projects/liuyixin/projects",tar=log_dir+"code/")


import os
gpu_ids = args.gpu
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_ids

# mnist = input_data.read_data_sets("mnist/", one_hot=True)
gpu_ids = gpu_ids.split(',')
gpu_ids = [int(id) for id in gpu_ids]
num_gpus = len(gpu_ids)
print("Available GPU Number :"+str(num_gpus))

# 读入数据
dataset = tf_dataset(batch_size=batch_size*num_gpus,epoch_size=None)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

def get_available_gpus():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
 
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']
def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device
 
    return _assign
 
def ft_resnet50(x,istrain=False):
    from tensorflow.contrib.slim import nets
    keep_rate = 1.0
    is_training = False
    if istrain:
        keep_rate = args.droprate
        is_training = True
    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        net, endpoints = nets.resnet_v1.resnet_v1_50(x, num_classes=None,
                                                        is_training=is_training,global_pool=True)
    import math
    with tf.variable_scope('Logits'):
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
    return logits,norm_feat


def get_varlist():
    checkpoint_exclude_scopes = 'Logits'
    exclusions = None
    if checkpoint_exclude_scopes:
        exclusions = [
            scope.strip() for scope in checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    add_var = []

    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
        if not excluded:
            variables_to_restore.append(var)
        else:
            add_var.append(var)
    return variables_to_restore,add_var


config = tf.ConfigProto(log_device_placement=True, allow_soft_placement = True)
config.gpu_options.allow_growth = True


def adjust_LR(step,epoch):
    lr_l = args.init_lr * 0.1**((epoch)//40)
    lr_b = 0.1 * lr_l
    if epoch < warmup:
        factor = min(1.0, 0.1 + 0.9 / (12936//(batch_size*num_gpus)*warmup))
        lr_l,lr_b = factor*lr_l,factor*lr_b
    return lr_l,lr_b

# def train():
with tf.device("/cpu:0"):
    global_step=tf.train.get_or_create_global_step()
    tower_grads = []

    x = tf.placeholder(tf.float32, [None, im_size[0],im_size[1],3],name='inputs')
    y_ = tf.placeholder(tf.int32, [None],name='labels')
    lr = tf.placeholder(tf.float32, name='lr')
    is_training = tf.placeholder(tf.bool, name='is_training')
    keep_rate = tf.placeholder(tf.float32, name='keep_rate')

    optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9,use_nesterov=True)
    # optimizer2 = tf.train.MomentumOptimizer(learning_rate=0.05,momentum=0.9,use_nesterov=True)
    
    with tf.variable_scope(tf.get_variable_scope()):
        for i in gpu_ids:
            with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
                _x = x[i * batch_size:(i + 1) * batch_size]
                _y = y_[i * batch_size:(i + 1) * batch_size]
                logits,norm_feat = ft_resnet50(_x,True)

                variables_to_restore,add_var = get_varlist()
                
                circle_loss_op,sp,sn,recall,precision,f1_score = PairWiseCircleLoss(norm_feat,_y) 
                softmaxloss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=_y, logits=logits)
                softmaxloss = tf.reduce_mean(softmaxloss)
                circle_loss_term = lamda * circle_loss_op/batch_size
                loss = softmaxloss + circle_loss_term
                grads = optimizer.compute_gradients(loss)
                # grads2 = optimizer2.compute_gradients(loss,var_list=add_var)
                tower_grads.append(grads)
                # tower_grads.append(grads2)
                softmaxlogits = tf.nn.softmax(logits)
                classes = tf.argmax(softmaxlogits, axis=1, name='classes')
                accuracy_softmax = tf.reduce_mean(tf.cast(
                    tf.equal(tf.cast(classes, dtype=tf.int32), _y), dtype=tf.float32))

                tf.summary.histogram('sn',sn)
                tf.summary.histogram('sp',sp)
                # recall,precision,f1_score
                tf.summary.scalar('pairwise_recall',recall)
                tf.summary.scalar('pairwise_precision',precision)
                tf.summary.scalar('pairwise_f1_score',f1_score)
                tf.summary.scalar('softmaxloss',softmaxloss)
                # tf.summary.scalar('reg_loss',reg_term)
                tf.summary.scalar('circleloss',circle_loss_term)
                tf.summary.scalar('totalloss',loss)
                tf.summary.scalar('accuracy_softmax',accuracy_softmax)
                tf.summary.scalar('lr',lr)
                merged = tf.summary.merge_all()
            grads = average_gradients(tower_grads)                                         
            train_op = optimizer.apply_gradients(grads)
            sample_counter=0
            with tf.Session(config=config) as sess:
                
                train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)

                sess.run(tf.global_variables_initializer())
                # print(variables_to_restore)
                saver_restore = tf.train.Saver(var_list=variables_to_restore)
                saver_restore.restore(sess, resnet_model_path)
                saver = tf.train.Saver(tf.global_variables())
                epoch = 0
                stepcounter = 0
                # for step in range(1, 50000):
                t0 = time.time()
                while epoch < epoch_size:
                    ts = time.time()
                    stepcounter+=1
                    images, groundtruth_lists = sess.run(one_element)
                    batch_now = len(images)
                    sample_counter+=batch_now
                    epoch = sample_counter // 12936
                    lr_l,lr_b = adjust_LR(stepcounter,epoch)
                    train_dict = {x: images, 
                        y_: groundtruth_lists,
                        is_training: True,
                        lr:lr_l,
                        keep_rate:args.droprate}
                    summary,_ = sess.run([merged,train_op], feed_dict=train_dict)
                    train_writer.add_summary(summary, stepcounter)
                    te = time.time() - ts
                    if stepcounter % 10 == 0 or stepcounter == 1:
                        loss_value, acc,f1 = sess.run([loss, accuracy_softmax,f1_score], feed_dict=train_dict)
                        # print(f"Epoch:{epoch} Step:" + str(stepcounter) + " loss:" + str(loss_value) + " softmaxAcc:" + str(acc)+f' f1score:{f1}'+", %i sample/sec" % int(batch_now/te))
                        timing = int(time.time()-t0)
                        print("Epoch:{} Step:{} TrainLoss:{:.4f} SoftmaxAcc:{:.4f} PairwiseF1:{:.4f} BatchRate:{}/sec Timing:{}m{}s".format(epoch,stepcounter,loss_value,acc,f1,int(batch_now/te),(timing)//60,timing%60))
                    if (epoch+1) % 10 == 0:
                        saver.save(sess, model_save_path, global_step=stepcounter+1)
                        print('save mode to {}'.format(model_save_path))
train_writer.close()