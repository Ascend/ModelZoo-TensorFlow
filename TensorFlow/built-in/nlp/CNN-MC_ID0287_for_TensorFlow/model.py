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
import tensorflow as tf
import numpy as np
from math import ceil
import data_utils
import sys

class CNN(object):
    def __init__(self, config, sess):
        self.n_epochs = config['n_epochs']
        self.kernel_sizes = config['kernel_sizes']
        self.n_filters = config['n_filters']
        self.dropout_rate = config['dropout_rate']
        self.val_split = config['val_split']
        self.edim = config['edim']
        self.n_words = config['n_words']
        self.std_dev = config['std_dev']
        self.input_len = config['sentence_len']
        self.batch_size = config['batch_size']
        self.inp = tf.placeholder(shape=[None, self.input_len], dtype='int32') #输入数据
        self.labels = tf.placeholder(shape=[None,], dtype='int32') #标签
        self.loss = None
        self.session = sess #回话
        self.cur_drop_rate = tf.placeholder(dtype='float32')

    def build_model(self):
        word_embedding = tf.Variable(tf.random_normal([self.n_words, self.edim], stddev=self.std_dev)) #正态分布词嵌入矩阵，维度为(n_words, 300, 1)
        # 通过tf.nn.embedding_lookup()得到每一个句子的表示（矩阵）,维度：(sen_len, 300, 1)
        # （详见https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow）
        x = tf.nn.embedding_lookup(word_embedding, self.inp)
        x_conv = tf.expand_dims(x, -1) #拓展维度，作为卷积的输入,因为卷积输入要求的四维
        
        #3个(3, 4, 5)卷积过滤器（权重），维度为：(3, 300, 1, 100)
        #详细可参考论文中的模型https://arxiv.org/pdf/1408.5882.pdf: filter windows (h) of 3, 4, 5 with 100 feature maps each,
        F1 = tf.Variable(tf.random_normal([self.kernel_sizes[0], self.edim ,1, self.n_filters] ,stddev=self.std_dev),dtype='float32')
        F2 = tf.Variable(tf.random_normal([self.kernel_sizes[1], self.edim, 1, self.n_filters] ,stddev=self.std_dev),dtype='float32')
        F3 = tf.Variable(tf.random_normal([self.kernel_sizes[2], self.edim, 1, self.n_filters] ,stddev=self.std_dev),dtype='float32')
        #3个卷积过滤器（权重）对应的偏差b
        FB1 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
        FB2 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
        FB3 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))

        #Weight for final layer
        W = tf.Variable(tf.random_normal([3*self.n_filters, 2], stddev=self.std_dev),dtype='float32')
        b = tf.Variable(tf.constant(0.1, shape=[1,2]),dtype='float32')

        #Convolutions层
        C1 = tf.add(tf.nn.conv2d(x_conv, F1, [1, 1, 1, 1], padding='VALID'), FB1) #WX + B
        C2 = tf.add(tf.nn.conv2d(x_conv, F2, [1, 1, 1, 1], padding='VALID'), FB2) #WX + B
        C3 = tf.add(tf.nn.conv2d(x_conv, F3, [1, 1, 1, 1], padding='VALID'), FB3) #WX + B
        C1 = tf.nn.relu(C1) #relu(WX + B)
        C2 = tf.nn.relu(C2) #relu(WX + B)
        C3 = tf.nn.relu(C3) #relu(WX + B)

        #Max pooling层
        maxC1 = tf.nn.max_pool(C1, [1, C1.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
        maxC1 = tf.squeeze(maxC1, [1, 2])
        maxC2 = tf.nn.max_pool(C2, [1, C2.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
        maxC2 = tf.squeeze(maxC2, [1, 2])
        maxC3 = tf.nn.max_pool(C3, [1, C3.get_shape()[1], 1, 1], [1, 1, 1, 1], padding='VALID')
        maxC3 = tf.squeeze(maxC3, [1, 2])

        #连接pooling层的特征
        z = tf.concat(axis=1, values=[maxC1, maxC2, maxC3]) #连接三个特征
        zd = tf.nn.dropout(z, self.cur_drop_rate) #Fully connected layer with dropout and softmax output

        #全连接层
        self.y = tf.add(tf.matmul(zd,W), b)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.labels) #softmax分类器损失函数，参数分别为

        self.loss = tf.reduce_mean(losses)
        self.optim = tf.train.AdamOptimizer(learning_rate=0.001) #Adam优化器
        self.train_op = self.optim.minimize(self.loss) #使用优化器最小化损失函数

    '''
    模型训练函数
    '''
    def train(self, data, labels):
        self.build_model() #整个模型定义过程，
        n_batches = int(ceil(data.shape[0]/self.batch_size))
        tf.global_variables_initializer().run() #
        t_data, t_labels, v_data, v_labels = data_utils.generate_split(data, labels, self.val_split) #划分数据为训练、验证集
        for epoch in range(1,self.n_epochs+1):
            train_cost = 0
            for batch in range(1,n_batches+1):
                import time
                start_time=time.time()
                X, y = data_utils.generate_batch(t_data, t_labels, self.batch_size) #每次抽取一个batch
                #将数据喂入placeholder
                f_dict = {
                    self.inp : X,
                    self.labels : y,
                    self.cur_drop_rate : self.dropout_rate
                }    

                _, cost = self.session.run([self.train_op, self.loss], feed_dict=f_dict) #得到每个batch的损失
                train_cost += cost #累计总损失
                # sys.stdout.write('Epoch %d Cost  :   %f - Batch %d of %d   \r' %(epoch, cost ,batch ,n_batches))
                # sys.stdout.flush()
                print('Train-Epoch %d Cost  :   %f - Batch %d of %d  - Train-time: %f   \r' % (epoch, cost, batch, n_batches, time.time() - start_time))

            print

            # BGN-New Add
            tf.train.Saver().save(self.session, "checkpoints/model.ckpt")
            tf.io.write_graph(self.session.graph, './checkpoints', 'graph.pbtxt', as_text=True)
            # END-New Add
            
            self.test(v_data, v_labels)
    
    def test(self,data,labels):
        n_batches = int(ceil(data.shape[0]/self.batch_size))
        test_cost = 0
        preds = []
        ys = []
        for batch in range(1,n_batches+1):
            import time
            start_time = time.time()
            X, Y = data_utils.generate_batch(data, labels, self.batch_size)
            f_dict = {
                self.inp : X,
                self.labels : Y,
                self.cur_drop_rate : 1.0 #测试阶段不使用droup out
            }    
            cost, y = self.session.run([self.loss,self.y], feed_dict=f_dict)
            test_cost += cost
            # sys.stdout.write('Cost  :   %f - Batch %d of %d   \r' %(cost ,batch ,n_batches))
            # sys.stdout.flush()
            #print('Test-Cost  :   %f - Batch %d of %d  - Test-time:  %f   \r' % (cost, batch, n_batches, time.time() - start_time))
            preds.extend(np.argmax(y,1))
            ys.extend(Y)

        print
        print("Accuracy", np.mean(np.asarray(np.equal(ys, preds), dtype='float32'))*100)