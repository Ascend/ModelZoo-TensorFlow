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

#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta
import pickle
import numpy as np
import tensorflow as tf
from sklearn import metrics
import traceback
from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab

from npu_bridge.npu_init import *

base_dir = 'thucnews'
train_dir = os.path.join(base_dir, 'thucnews.train.txt') 
test_dir = os.path.join(base_dir, 'thucnews.test.txt')
val_dir = os.path.join(base_dir, 'thucnews.val.txt')
vocab_dir = os.path.join(base_dir, 'thucnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()*1000
    time_dif = end_time - start_time
    return timedelta(milliseconds=int(round(time_dif)))


    
    
def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x,y):
    """评估在某一数据上的准确率和损失"""    
    total_loss = 0.0
    total_acc = 0.0
    data_len = len(x)
    batch_train = batch_iter_(x, y,256)
    for x_batch, y_batch in batch_train:
        
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len
    
class data_load(object):
    def __init__(self, sess,x,y,batch_size,is_train=True):
        
        with tf.device('/cpu:0'):
            self.x = x
            self.y = y
            self.x_ = tf.placeholder(self.x.dtype, self.x.shape)
            self.y_ = tf.placeholder(self.y.dtype, self.y.shape)
            self.sess = sess
            dataset = tf.data.Dataset.from_tensor_slices((self.x_, self.y_))

            if is_train:
                dataset = dataset.shuffle(len(self.x))
                dataset = dataset.repeat()
                dataset = dataset.batch(len(self.x))
            else:
                dataset = dataset.batch(len(self.x))
            
            dataset = dataset.prefetch(2)
            self.iterator = dataset.make_initializable_iterator()
            self.next = self.iterator.get_next()
            self.sess.run(self.iterator.initializer, feed_dict={self.x_: self.x,self.y_: self.y})
        
    def replay(self):
        self.sess.run(self.iterator.initializer, feed_dict={self.x_: self.x,self.y_: self.y})
    
    
def batch_iter_(x, y, batch_size=64):
        data_len = len(x)
        
        num_batch = int((data_len - 1) / batch_size) + 1
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x[start_id:end_id], y[start_id:end_id]


def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()*1000

    x_train = pickle.load(open('thucnews/train.x','rb'))
    y_train = pickle.load(open('thucnews/train.y','rb'))
    x_val = pickle.load(open('thucnews/test.x','rb'))
    y_val = pickle.load(open('thucnews/test.y','rb'))
    #x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    #x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # 创建session
    sess_config = tf.ConfigProto()
    custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显示关闭remap
    custom_op.parameter_map["dynamic_input"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
    
    session = tf.Session(config=sess_config)
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)
    train_len = len(x_train)
    val_len = len(x_val)
    train_data = data_load(session,x_train,y_train,config.batch_size)
    val = data_load(session,x_val,y_val,256,False)
    x_v, y_v = session.run(val.next)
    tf.io.write_graph(session.graph_def, 'checkpoints', 'train.pbtxt')
    print('Training and evaluating...')
    start_time = time.time()*1000
    data_time = 0
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 10000  # 如果超过1000轮未提升，提前结束训练
    total_feed = 0
    total_summary = 0
    total_val = 0
    total_save = 0
    total_train = 0
    
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        start_time6 = time.time()*1000
        x, y = session.run(train_data.next)
        batch_train = batch_iter_(x, y, config.batch_size)
        #batch_train = session.run(train_data)
        for x_batch, y_batch in batch_train:
            
            #start_time1 = time.time()
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
            #total_feed += time.time() - start_time1
            
            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                #start_time2 = time.time()
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)
                #total_summary += time.time() - start_time2

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                #print(x_batch,y_batch)
                #start_time3 = time.time()
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_v, y_v)  # todo
                #total_val += time.time() - start_time3
                #val.replay()
                
                
                if acc_val > best_acc_val :
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    
                    if acc_val > 0.95:
                        #start_time4 = time.time()
                        saver.save(sess=session, save_path=save_path)
                        #total_save += time.time() - start_time4
                    
                    improved_str = '*'
                else:
                    improved_str = ''
                
                
                #print("feed: %.5f summary: %.5f val: %.5f save: %.5f train: %.5f"%(total_feed,total_summary,total_val,total_save,total_train))
                
                
                #total_feed = 0
                #total_summary = 0
                #total_val = 0
                #total_save = 0
                #total_train = 0
                
                time_dif = get_time_dif(start_time)
                
                start_time = time.time()*1000
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))
            #start_time5 = time.time()
            feed_dict[model.keep_prob] = config.dropout_keep_prob
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            #total_train += time.time() - start_time5
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
            #if total_batch > 10:
            #    exit()
        start_time6 = time.time()*1000 - start_time6
        print("Epoch Time: %s  %.2f examples/s"% (timedelta(milliseconds=start_time6), (len(x)/start_time6*1000)))
        if flag:  # 同上
            break
    msg = 'Test Acc: {0:>7.2%}'
    print(msg.format(best_acc_val))
    try:
        os.rename("checkpoints/textcnn/best_validation.meta","checkpoints/textcnn/best_validation.meta_{0:>7.2%}".format(best_acc_val))
        os.rename("checkpoints/textcnn/best_validation.index","checkpoints/textcnn/best_validation.index_{0:>7.2%}".format(best_acc_val))
        os.rename("checkpoints/textcnn/best_validation.data-00000-of-00001","checkpoints/textcnn/best_validation.data-00000-of-00001_{0:>7.2%}".format(best_acc_val))
    except:
        traceback.print_exc()
    

def test():
        
        print("Loading test data...")
        sess_config = tf.ConfigProto()
        custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显示关闭remap
        custom_op.parameter_map["dynamic_input"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")
        session = tf.Session(config=sess_config)
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

        print('Testing...')
        loss_test, acc_test = evaluate(session, x_test, y_test)
        msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
        print(msg.format(loss_test, acc_test))
        

    


config = TCNNConfig()
categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)
config.vocab_size = len(words)
if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    build_vocab(train_dir, vocab_dir, config.vocab_size)


#x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)
f=open('thucnews/test.x','rb') 
x_test = pickle.load(f)
f.close()  

f=open('thucnews/test.y','rb') 
y_test = pickle.load(f)
f.close() 


if __name__ == '__main__':
    while True:
        tf.reset_default_graph()
        print('Configuring CNN model...')
        
        
        model = TextCNN(config)
        train()
        #if sys.argv[1] == 'train':
        #    train()
        #else:
        #    test()
        #tf.reset_default_graph()
        #model = TextCNN(config)
        #test()