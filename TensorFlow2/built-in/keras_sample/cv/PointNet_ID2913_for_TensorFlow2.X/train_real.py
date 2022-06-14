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
import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=5, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 4096
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data_real/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data_real/test_files.txt'))
print(TRAIN_FILES)
print(TEST_FILES)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


# 计算指数衰减的学习率。训练时学习率最好随着训练衰减。
# tf.train.exponential_decay函数实现指数衰减学习率。
def get_learning_rate(batch):
    # 在Tensorflow中，为解决设定学习率(learning rate)问题，提供了指数衰减法来解决。
    # 通过tf.train.exponential_decay函数实现指数衰减学习率。
    # 学习率较大容易搜索震荡（在最优值附近徘徊），学习率较小则收敛速度较慢，
    # 那么可以通过初始定义一个较大的学习率，通过设置decay_rate来缩小学习率，减少迭代次数。
    # tf.train.exponential_decay就是用来实现这个功能。
    #
    # 步骤：
    # 1.首先使用较大学习率(目的：为快速得到一个比较优的解);
    # 2.然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定);
    learning_rate = tf.compat.v1.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    # 训练时学习率最好随着训练衰减，learning_rate最大取0.00001 (衰减后的学习率和0.00001取最大)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        


# 取得bn衰减
# if the argument staircase is True,
# then global_step /decay_steps is an integer division and the decayed learning rate follows a staircase function.
# 计算衰减的Batch Normalization 的 decay。
def get_bn_decay(batch):
    # 指数衰减法

    # 在Tensorflow中，为解决设定学习率(learning rate)问题，提供了指数衰减法来解决。
    # 通过tf.train.exponential_decay函数实现指数衰减学习率。
    # 学习率较大容易搜索震荡（在最优值附近徘徊），学习率较小则收敛速度较慢，
    # 那么可以通过初始定义一个较大的学习率，通过设置decay_rate来缩小学习率，减少迭代次数。
    # tf.train.exponential_decay就是用来实现这个功能。
    #
    # 步骤：
    # 1.首先使用较大学习率(目的：为快速得到一个比较优的解);
    # 2.然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定);
    bn_momentum = tf.compat.v1.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    # bn衰减0.99和1-衰减后的动量，取最小
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


# 初始运行的训练函数。
# 这一段主要是通过placeholder进行赋值, 模型的参数准备和构建整个训练网络(数据处理+loss+优化器),模型记录工作,最后进行训练.
def train():
    # 将这个类实例，也就是新生成的图作为整个 tensorflow 运行环境的默认图
    with tf.Graph().as_default():
        # 如果需要切换成CPU运算，可以调用tf.device(device_name)函数，其中device_name格式如 /cpu:0 其中的0表示设备号，
        # TF不区分CPU的设备号，设置为0即可。GPU区分设备号 /gpu:0 和 /gpu:1 表示两张不同的显卡。
        # with tf.device('/gpu:'+str(GPU_INDEX)):
        with tf.device('/cpu:0'):
            # 使用了pointne_cls.py的placeholder_inputs（）方法。
            # 取得占位符，点云，标签。 输入是 一批数据的数量，点的数量。
            # placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，
            # 它只会分配必要的内存，用于传入外部数据。
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            # 向指定好的对象中喂入数据：tf.placeholder()
            # 取得占位符：是否在训练。
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            # 将 global_step = batch 参数最小化。
            # 这是在告诉优化器 在每次训练时 为你有用地增加'batch'参数。
            # 定义 batch = 0
            batch = tf.Variable(0)
            # 取得bn衰减（自定义方法）
            bn_decay = get_bn_decay(batch)
            # 用来显示标量信息，一般在画loss,accuary时会用到这个函数。
            tf.compat.v1.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            # 创建的数据处理网络为pred，调用 model\pointnet_cls 下的get_model()得到。由get_model()可知，
            # pred的维度为B×N×40，40为分出的类别Channel数，对应40个分类标签。每个点的这40个值最大的一个的下标即为所预测的分类标签。
            # 首先使用共享参数的MLP对每个点进行特征提取，再使用MaxPooling在特征维进行池化操作，
            # 使得网络对不同数量点的点云产生相同维度的特征向量，且输出对输入点的顺序产生不变性。
            # 在得到固定维度的特征向量之后，再使用一个MLP对其进行分类。
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            # 调用pointnet_cls下的get_loss（）
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.compat.v1.summary.scalar('loss', loss)

            # tf.argmax(pred, 2) 返回pred C 这个维度的最大值索引返回相同维度的bool值矩阵
            # tf.equal() 比较两个张量对应位置是否相等
            correct = tf.equal(tf.argmax(input=pred, axis=1), tf.cast(labels_pl, dtype=tf.int64))
            # 压缩求和，用于降维
            accuracy = tf.reduce_sum(input_tensor=tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.compat.v1.summary.scalar('accuracy', accuracy)

            # Get training operator
            # 取得学习率（自定义方法），获得衰减后的学习率，以及选择优化器optimizer。
            learning_rate = get_learning_rate(batch)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            # minimize的内部存在两个操作：(1)计算各个变量的梯度 (2)用梯度更新这些变量的值
            # (1)计算loss对指定val_list的梯度（导数），返回元组列表[(gradient,variable),…]
            # (2)用计算得到的梯度来更新对应的变量（权重）
            # 注意：在程序中global_step初始化为0，每次更新参数时，自动加1
            # 将minimize()分成两个步骤的原因：在某种情况下对梯度进行修正，防止梯度消失或者梯度爆炸
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.compat.v1.train.Saver()
            
        # Create a session
        # 配置session 运行参数。
        # 创建sess的时候对sess进行参数配置
        config = tf.compat.v1.ConfigProto()
        # =True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
        config.gpu_options.allow_growth = True
        # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
        config.allow_soft_placement = True
        # 在终端打印出各项操作是在哪个设备上运行的
        config.log_device_placement = False
        # 创建 sess, 才能运行框架
        sess = tf.compat.v1.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        # 初始化参数，开始训练
        # train_one_epoch 函数用来训练一个epoch，eval_one_epoch函数用来每运行一个epoch后evaluate在测试集的
        # accuracy和loss。每10个epoch保存1次模型。
        init = tf.compat.v1.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        # 运行sess初始化所有的全局变量
        sess.run(init, {is_training_pl: True})

        # ops 是一个字典，作为接口传入训练和评估 epoch 循环中。
        # pred 是数据处理网络模块；loss 是 损失函数；train_op 是优化器；batch 是当前的批次
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            # log（自定义方法）
            log_string('**** EPOCH %03d ****' % (epoch))
            # 在同一个位置刷新输出
            sys.stdout.flush()

            # 训练一个批次（自定义方法）
            # train_one_epoch 函数用来训练一个epoch
            train_one_epoch(sess, ops, train_writer)
            # 评估一个批次（自定义方法）
            # eval_one_epoch函数用来每运行一个epoch后evaluate在测试集的accuracy和loss
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            # Save the variables to disk.每10个epoch保存1次模型
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                # log（自定义方法）
                log_string("Model saved in file: %s" % save_path)


# provider.shuffle_data 函数随机打乱数据，返回打乱后的数据。
# num_batches = file_size/BATCH_SIZE，计算在指定BATCH_SIZE下，训练1个epoch 需要几个mini-batch训练。
def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train files
    # 随机打乱训练数据
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)
    
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0

        # 在一个epoch 中逐个mini-batch训练直至遍历完一遍训练集。计算总分类正确数total_correct和已遍历样本数
        # total_senn，总损失loss_sum.
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            
            # Augment batched point clouds by rotation and jittering
            # 调用provider中rotate_point_cloud
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            # 训练，使用 tf 的 session 运行设计的框架，ops['pred'] 为整个网络，feed_dict 为网络提供的数据
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val

        # 记录平均loss，以及平均accuracy。
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
         


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
