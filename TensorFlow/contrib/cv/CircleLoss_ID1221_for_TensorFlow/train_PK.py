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
import numpy as np
import skimage.transform
import tensorflow as tf
# from config import *
from model import *
from data import *
from loss import *
from tensorflow.contrib.slim import nets
import time 
import argparse


arg = argparse.ArgumentParser()
timestamp = int(time.time())
arg.add_argument('-e','--expriment_name',type=str,default=f'test_{timestamp}')
arg.add_argument('-g','--gpu',type=str,default='0,1')
arg.add_argument('-w','--warmup',type=int,default=5)
# arg.add_argument('-b','--batch_size',type=int,default=32)
arg.add_argument('-epoch','--epoch_size',type=int,default=60)
arg.add_argument('-modelname','--modelname',type=str,default='softmaxcircleloss')
arg.add_argument('-lr','--init_lr',type=float,default=0.02)
arg.add_argument('-lamda','--lamda',type=float,default=1)
arg.add_argument('-drop','--droprate',type=float,default=0.5)
arg.add_argument("--train_url", type=str, default="./output")
arg.add_argument("--data_url", type=str, default="./dataset")
arg.add_argument("--ckpt_url", type=str, default="./ckpt")
arg.add_argument("--erasing_prob", type=float, default=0.5)
arg.add_argument("--P", type=int, default=4)
arg.add_argument("--K", type=int, default=4)


args = arg.parse_args()


homedir = "/home/nanshen/xutan/yixin/circlesoftmax"
log_dir = f'{homedir}/logs/{args.expriment_name}/'
im_size = [256, 128]
market_dir = 'Data/Market-1501-v15.09.15'
warmup = args.warmup
m=0.25
gamma=256
lamda = args.lamda
num_classes=751
epoch_size = args.epoch_size
K=args.K
P=args.P
batch_size = K*P
embedding_size=128
resnet_model_path = '/home/nanshen/xutan/yixin/circlesoftmax/checkpoint/resnet_v2_50.ckpt'  # Path to the pretrained model
model_save_path = log_dir + args.modelname  # Path to the model.ckpt-(num_steps) will be saved
num_bottleneck = 512
init_lr = args.init_lr
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu


x = tf.placeholder(tf.float32, [None, im_size[0],im_size[1],3],name='inputs')
y_ = tf.placeholder(tf.int32, [None],name='labels')
lr_backbone = tf.placeholder(tf.float32, name='lr_backbone')
lr_logit = tf.placeholder(tf.float32, name='lr_logit')
is_training = tf.placeholder(tf.bool, name='is_training')
keep_rate = tf.placeholder(tf.float32, name='keep_rate')


# 读入数据
imnum, datasets,le = tf_dataset_PK(P=args.P,K=args.K,erasing_prob=args.erasing_prob,epoch_size=None,path='/home/nanshen/xutan/yixin/market1501/Market-1501-v15.09.15/bounding_box_train')
one_elements=[]
for i in range(751):
    iterator = datasets[i].make_one_shot_iterator()
    one_element = iterator.get_next()
    one_elements.append(one_element)

def get_PKbatch(sess,one_elements,P=4,K=4):
    random.shuffle(one_elements)
    this_batch = one_elements[:P]
    images,labels = 0,0
    for i in range(P):
        if i == 0:
            images ,labels = sess.run(this_batch[i])
        else:
            image,label = sess.run(this_batch[i])
            images,labels = tf.concat([images,image],0),tf.concat([labels,label],0)
    images,labels = sess.run([images,labels])
    return images,labels
# with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
#     net, endpoints = nets.resnet_v1.resnet_v1_50(x, num_classes=None,
#                                                     is_training=is_training,global_pool=True)
# from tf_slim.nets import resnet_v2
with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
      net, end_points = nets.resnet_v2.resnet_v2_50(x, num_classes=None, is_training=is_training)

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
        circle_loss_op,sp,sn,recall,precision,f1_score = PairWiseCircleLoss(norm_feat,y_)   
        tf.summary.histogram('sn',sn)
        tf.summary.histogram('sp',sp)
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
# regularizer = tf.contrib.layers.l2_regularizer(5e-4)
# reg_term = tf.contrib.layers.apply_regularization(regularizer,weights_list=add_var)

# softmaxloss = tf.nn.sparse_softmax_cross_entropy_with_logits(
#     labels=y_, logits=logits)
# softmaxloss = tf.reduce_mean(softmaxloss)
# soft_logit = tf.clip_by_value(soft_logit, 0.000001, 0.999999)
labels = tf.one_hot(y_,num_classes)
soft_logit = tf.nn.softmax(logits)
# softmaxloss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
soft_logits = tf.clip_by_value(soft_logit, 0.000001, 0.999999)
cross_entropy = -tf.reduce_sum(labels * tf.log(soft_logits), axis=1)
softmaxloss = tf.reduce_mean(cross_entropy)
circle_loss_term = lamda * circle_loss_op/batch_size
loss = softmaxloss + circle_loss_term 


soft_logit = tf.nn.softmax(logits)
classes = tf.argmax(soft_logit, axis=1, name='classes')
accuracy_softmax = tf.reduce_mean(tf.cast(
    tf.equal(tf.cast(classes, dtype=tf.int32), y_), dtype=tf.float32))

# optimizer1 = tf.train.AdamOptimizer(learning_rate=lr_backbone)
# optimizer2 = tf.train.AdamOptimizer(learning_rate=lr_logit)
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

config = tf.ConfigProto(allow_soft_placement = True) 
config.gpu_options.allow_growth = True      #程序按需申请内存

# tensorboard记录下来

# recall,precision,f1_score
tf.summary.scalar('pairwise_recall',recall)
tf.summary.scalar('pairwise_precision',precision)
tf.summary.scalar('pairwise_f1_score',f1_score)
tf.summary.scalar('softmaxloss',softmaxloss)
# tf.summary.scalar('reg_loss',reg_term)
tf.summary.scalar('circleloss',circle_loss_term)
tf.summary.scalar('totalloss',loss)
tf.summary.scalar('accuracy_softmax',accuracy_softmax)
tf.summary.scalar('lr_backbone',lr_backbone)
tf.summary.scalar('lr_logit',lr_logit)
merged = tf.summary.merge_all()

def adjust_LR(step,epoch):
    lr_l = init_lr * 0.1**(epoch//40)
    # if epoch <= 3:
    #     lr_b = 0
    # else:
    #     lr_b = 0.1 * lr_l
    lr_b = 0.1 * lr_l
    if epoch < warmup:
        factor = min(1.0, 0.1 + 0.9 / (imnum//batch_size*warmup))
        lr_l,lr_b = factor*lr_l,factor*lr_b

    if step < 2000: # freeze bone trick
        lr_b = 0

    return lr_l,lr_b

# 开始训练
with tf.Session(config=config) as sess:
    
    # writter = tf.summary.FileWriter(log_dir, sess.graph)
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    # test_writer = tf.summary.FileWriter(log_dir + '/test',sess.graph)

    # 执行初始化操作
    sess.run(init)
    
    # Load the pretrained checkpoint file xxx.ckpt
    saver_restore.restore(sess, resnet_model_path)

    # for i in range(step_size):
        # if i % 500 ==0:
        #     saver.save(sess, log_dir + "/model.ckpt", i)
        # if i % 10 == 0:
        #     # 每10步测试一下
        #     summary = sess.run(merged, feed_dict=feed_dict(marketdata,False,x,y_,lr,lr_new))
        #     test_writer.add_summary(summary, i)
        # else:
        #     if i % 100 == 99:
        #         # 每100次记录训练运算时间和内存占用等信息
        #         run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #         run_metadata = tf.RunMetadata()
        #         summary, _ = sess.run([merged, train_step],
        #                             feed_dict=feed_dict(marketdata,True,x,y_,lr,lr_new),
        #                             options=run_options,
        #                             run_metadata=run_metadata)
        #         train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        #         train_writer.add_summary(summary, i)
        #     else:
        #         # 正常情况只进行汇总和训练步伐操作
        #         summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(marketdata,True,x,y_,lr,lr_new))
        #         train_writer.add_summary(summary, i)
    epoch = 0
    sample_counter = 0
    step_counter = 0
    # for i in range(step_size):
    import time
    t0=time.time()
    while epoch < epoch_size:
        step_counter+=1
        # epoch = np.mean(traindata._epochs_completed)
        import time
        ts = time.time()
        # images, groundtruth_lists = sess.run(one_element)
        images, groundtruth_lists = get_PKbatch(sess,one_elements,P=args.P,K=args.K)
        # images_test, groundtruth_lists_test = sess.run(testnext)

        # print(groundtruth_lists)
        batch_now = int(images.shape[0])
        sample_counter+=batch_now
        epoch = sample_counter // imnum
        # if batch_now < batch_size: # skip the last batch
        #     print('skip last')
        #     continue
        lr_l,lr_b = adjust_LR(step_counter,epoch)

        train_dict = {x: images, 
                        y_: groundtruth_lists,
                        is_training: True,
                        lr_logit:lr_l,
                        lr_backbone:lr_b,
                        keep_rate:0.5}
        # test_dict = {x: images_test, 
        #                 y_: groundtruth_lists_test,
        #                 is_training: False,
        #                 lr_logit:lr_l,
        #                 lr_backbone:lr_b,
        #                 keep_rate:1.0}                
        summary,loss_, acc_,f1,softmaxlossterm,circlelossterm,_,_ = sess.run([merged,loss, accuracy_softmax,f1_score, softmaxloss,circle_loss_term,train_step1,train_step2], feed_dict=train_dict)
        # summary_test = sess.run(merged, feed_dict=test_dict)
        train_writer.add_summary(summary, step_counter)
        # test_writer.add_summary(summary_test,step_counter)
        
        t1 = time.time()
        timingperstep = (t1-ts)
        timing = int(t1 - t0)
        # train_text = 'Epoch:{}, Step: {}, train_Loss: {:.4f}, train_Accuracy: {:.4f}, Timing:{:.4f}s'.format(
        #     epoch,step_counter+1, loss_, acc_,t1-t0)
        print("Epoch:{} Step:{} TrainLoss:{:.4f} SoftmaxLoss:{:.4f} CircleLoss:{:.4f} SoftmaxAcc:{:.4f} PairwiseF1:{:.4f} TimePerStep:{:.4f}s TotalTiming:{}m{}s".format(epoch,step_counter,loss_,softmaxlossterm,circlelossterm,acc_,f1,timingperstep,(timing)//60,timing%60))
        
        if (step_counter+1) % 500 == 0:
            saver.save(sess, model_save_path+'.ckpt', global_step=step_counter+1)
            tf.io.write_graph(sess.graph, model_save_path, 'graph.pbtxt', as_text=True)
            print('save mode to {}'.format(model_save_path))

        # if (i+1) % 10 == 0:
        #     t0=time.time()
        #     images, groundtruth_lists = get_next_batch(marketdata,train=False)
        #     test_dict = {x: images, 
        #                 y_: groundtruth_lists,
        #                 is_training: True,
        #                 lr:lr_new}
        #     summary,loss_val,acc_val = sess.run([merged,loss, accuracy], feed_dict=test_dict)
        #     test_writer.add_summary(summary, i)
        #     t1 = time.time()
        #     test_text = 'Step: {}, Loss: {:.4f}, Accuracy: {:.4f}, Timing:{:.4f}s'.format(
        #     i+1, loss_val, acc_val,t1-t0)
        #     print(test_text)

train_writer.close()

os.system('python /home/nanshen/xutan/yixin/circlesoftmax/code/test.py')