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
from npu_bridge.npu_init import *
import tensorflow as tf
import numpy as np
import math

import utils
from model.linear_layer import Linear, RegLinear, DropLinear
import time

#########################################################################################

class TransferNN(object):
    def __init__(self, node_info, optim=('Adam',1e-4), name='[tf/NN]', trainable_info=None, keep_prob_info=None):

        self.name = name
        self.optim = optim
        self.node_info = node_info

        if keep_prob_info == None:
            keep_prob_info = [0.8] + [0.5] * (len(node_info) - 2)
        self.keep_prob_info = np.array(keep_prob_info)
        self.eval_keep_prob_info = np.array([1.0] * (len(node_info) - 1))

        if trainable_info == None:
            trainable_info = [True] * len(node_info)
        self.trainable_info = trainable_info

        self.x = tf.placeholder(tf.float32, shape=[None, np.prod(node_info[0])])
        self.y_ = tf.placeholder(tf.float32, shape=[None, node_info[-1]])
        self.drop_rate = tf.placeholder(tf.float32, shape=[len(node_info) - 1])

        self._BuildModel()
        self._CrossEntropyPackage(optim)


    def _BuildModel(self):
        h_out_prev = npu_ops.dropout(self.x, self.drop_rate[0])

        self.Layers = []
        self.Layers_dropbase = []
        for l in range(1, len(self.node_info)-1):
            self.Layers.append(DropLinear(h_out_prev, self.node_info[l], self.drop_rate[l]))
            self.Layers_dropbase.append(self.Layers[-1].dropbase)
            
            h_out_prev = tf.nn.relu(self.Layers[-1].h_out)

        self.Layers.append(DropLinear(h_out_prev, self.node_info[-1], 1.0))
        self.Layers_dropbase.append(self.Layers[-1].dropbase)
        self.y = self.Layers[-1].h_out

    def npu_tf_optimizer(self, opt):
        loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=100, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        npu_opt = NPULossScaleOptimizer(opt, loss_scale_manager)
        return npu_opt

    def _OptimizerPackage(self, obj, optim):
        if optim[0] == 'Adam': return self.npu_tf_optimizer(tf.train.AdamOptimizer(optim[1])).minimize(obj)
        elif optim[0] == 'SGD': return self.npu_tf_optimizer(tf.train.GradientDescentOptimizer(optim[1])).minimize(obj)
        elif optim[0] == 'Momentum': return self.npu_tf_optimizer(tf.train.MomentumOptimizer(optim[1][0],optim[1][1])).minimize(obj)

    def _CrossEntropyPackage(self, optim):
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        self.train_step = self._OptimizerPackage(self.cross_entropy, optim) 
        self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def RegPatch(self, delta):
        self.reg_obj = 0
        self.Layers_reg = []

        for l in range(0,len(self.Layers)):
            self.Layers_reg.append(RegLinear(self.Layers[l]))
            self.reg_obj += delta * self.Layers_reg[l].reg_obj

        cel = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y)
        self.cross_entropy = tf.reduce_mean(cel) + self.reg_obj
        self.train_step = self._OptimizerPackage(self.cross_entropy, self.optim)

    #def CalculateFisherMatrix(self, sess, x, y, mb=1000):
    def CalculateFisherMatrix(self, sess, x, y, mb=10000):
        """new version of Calculating Fisher Matrix    

        Returns:
            FM: consist of [FisherMatrix(layer)] including W and b.
                and Fisher Matrix is dictionary of numpy array
                i.e. Fs[idxOfLayer]['W' or 'b'] -> numpy
        """
        FM = []
        data_size = x.shape[0]
        total_step = int(math.ceil(float(data_size)/mb))

        for step in range(total_step):
            ist = (step * mb) % data_size
            ied = min(ist + mb, data_size)
            y_sample = tf.reshape(tf.one_hot(tf.multinomial(self.y, 1), 10), [-1, 10])
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_sample, logits=self.y))
            for l in range(len(self.Layers)):
                if len(FM) < l + 1:
                    FM.append({})
                    FM[l]['W'] = np.zeros(self.Layers[l].W.get_shape().as_list())
                    FM[l]['b'] = np.zeros(self.Layers[l].b.get_shape().as_list())
                W_grad = tf.reduce_sum(tf.square(tf.gradients(cross_entropy,[self.Layers[l].W])), 0)
                b_grad = tf.reduce_sum(tf.square(tf.gradients(cross_entropy,[self.Layers[l].b])), 0)
                W_grad_val, b_grad_val = sess.run([W_grad, b_grad],
                                    feed_dict={ self.x:x[ist:ied],
                                                self.drop_rate:self.eval_keep_prob_info})
                FM[l]['W'] += W_grad_val
                FM[l]['b'] += b_grad_val

        for l in range(len(self.Layers)):
            FM[l]['W'] += 1e-8
            FM[l]['b'] += 1e-8
        
        return FM

    def Train(self, sess, x, y, x_, y_, epoch, mb=50):
        ########## dataset #############
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.repeat(epoch)
        dataset = dataset.batch(mb)
        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        iterator = dataset.make_initializable_iterator()
        nextelement = iterator.get_next()
        sess.run(iterator.initializer)
        ########## dataset #############

        data_size = x.shape[0]
        total_step = int(math.ceil(float(data_size)/mb))

        for e in range(epoch):
            train_acc = 0
            train_loss = 0
            cost_time = 0
            for step in range(total_step):
                start_time = time.time()
                ist = (step * mb) % data_size
                ied = min(ist + mb, data_size)
                #### dataset #########
                (x, y) = sess.run(nextelement)
                _, acc, loss = sess.run([self.train_step, self.accuracy, self.cross_entropy],
                                    feed_dict={ self.x:x,
                                                self.y_:y,
                                                self.drop_rate:self.keep_prob_info})

                #### dataset #########
               # _, acc, loss = sess.run([self.train_step, self.accuracy, self.cross_entropy], 
               #                     feed_dict={ self.x:x[ist:ied],
               #                                 self.y_:y[ist:ied],
               #                                 self.drop_rate:self.keep_prob_info})

                train_acc += (ied - ist) * acc
                train_loss += (ied - ist) * loss
                cost_time += (time.time()-start_time)
            train_acc /= data_size
            train_loss /= data_size

            test_acc = self.Test(sess, [[x_,y_,""]], 1000, False)[0]
            print("(epoch = %d, epoch*total_step = %d, epoch*data_size = %d, train_acc = %.4f, test_acc = %.4f, train_loss = %.4f, avg_time = %.4f )" % (e+1, (e+1)*total_step,
                (e+1)*data_size, train_acc, test_acc, train_loss, cost_time/total_step))

    def Test(self, sess, xyc_info, mb=1000, debug=True): #ti: triple_info
        acc_ret = []

        for l in range(len(xyc_info)):
            x_, y_, c = xyc_info[l]
            comment = self.name + c
            acc = self._Test(sess, x_, y_, mb)
            acc_ret.append(acc)

            if debug: 
                print("%s accuracy : %.4f" % (comment, acc))

        return acc_ret

    def _Test(self, sess, x_, y_, mb):
        ########## dataset #############
        dataset = tf.data.Dataset.from_tensor_slices((x_, y_))
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(mb)
        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
        iterator = dataset.make_initializable_iterator()
        nextelement = iterator.get_next()
        sess.run(iterator.initializer)
        ########## dataset #############
        acc = 0
        data_size = x_.shape[0]
        for step in range(int(math.ceil(float(data_size)/mb))):
            ist = (step * mb) % data_size
            ied = min(ist + mb, data_size)
            ########## dataset #############
            (x_, y_) = sess.run(nextelement)
            acc += (ied - ist) * sess.run(self.accuracy,
                                feed_dict={ self.x:x_,
                                            self.y_:y_,
                                            self.drop_rate:self.eval_keep_prob_info})
            ########## dataset #############
            #acc += (ied - ist) * sess.run(self.accuracy, 
            #                    feed_dict={ self.x:x_[ist:ied], 
            #                                self.y_:y_[ist:ied], 
            #                                self.drop_rate:self.eval_keep_prob_info})
        acc /= data_size
        return acc


    def TestTasks(self, sess, x, y, x_, y_, mb=1000, debug=True): #ti: triple_info
        """
        test tasks using x, y, x_, y_ data.

        Args:
            x: list of original and shuffled input training data
            y: label of training image
            x_: list of original and shuffled input test data
                (the size should be same with the size of x)
            y_: label of test image
        Returns:
            ret: list of accuracy
                [training_accuracies, ..., test_accuracies, ...]
        """
        xyc_info = []
        for i in range(len(x)):
            xyc_info.append([x[i], y[i], 'train-idx%d' % i])
        for i in range(len(x_)):
            xyc_info.append([x_[i], y_[i], 'test-idx%d' % i])
     
        return self.Test(sess, xyc_info, mb=mb, debug=debug)

    def TestAllTasks(self, sess, x_tasks, y_tasks, mb=1000, debug=True): #ti: triple_info
        acc_ret = []
        for l in range(len(x_tasks)):
            x_ = x_tasks[l]
            y_ = y_tasks[l]
            acc = self._Test(sess, x_, y_, mb)
            acc_ret.append(acc)
        print(acc_ret)
        if debug: 
            print("%s all test accuracy : %.4f" % (self.name, np.average(acc_ret)))

        return acc_ret

