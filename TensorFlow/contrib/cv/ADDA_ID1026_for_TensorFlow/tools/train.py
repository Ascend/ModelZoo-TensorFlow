"""
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
"""
import logging
import os
import random
import sys
from collections import deque

import time
import click
import numpy as np
import tensorflow as tf

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import adda
from adda.data.mnist import MNIST,MNIST2000
from adda.data.usps import USPS, USPS1800
from adda.data.svhn import SVHN
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
from npu_bridge.npu_init import *
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

@click.command()
@click.argument('dataset')
@click.argument('split')
@click.argument('model')
@click.argument('output')
@click.option('--gpu', default='0')
@click.option('--iterations', default=20000)
@click.option('--batch_size', default=50)
@click.option('--display', default=10)
@click.option('--lr', default=1e-4)
@click.option('--stepsize', type=int)
@click.option('--snapshot', default=5000)
@click.option('--weights')
@click.option('--weights_end')
@click.option('--ignore_label', type=int)
@click.option('--solver', default='sgd')
@click.option('--seed',default=0 ,type=int)



def main(dataset, split, model, output, gpu, iterations, batch_size, display,
         lr, stepsize, snapshot, weights, weights_end, ignore_label, solver,
         seed):
    adda.util.config_logging()
    random.seed(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    dataset_name = dataset
    dataset = getattr(adda.data.get_dataset(dataset), split)
    model_fn = adda.models.get_model_fn(model)
    im_batch, label_batch = dataset.tf_ops_data(iterations, batch_size)
    im_batch_p = tf.placeholder(tf.float32, [None, 28, 28, 1])
    label_batch_p = tf.placeholder(tf.int32, [None])
    # label_batch = tf.cast(label_batch, tf.int32)
    net, layers = model_fn(im_batch_p)
    class_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(label_batch_p, net)
    loss = tf.compat.v1.losses.get_total_loss()
    # accuracy
    correct_prediction = tf.equal(tf.argmax(net,1), tf.cast(label_batch_p,tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    if solver == 'sgd':
        optimizer = tf.compat.v1.train.MomentumOptimizer(lr_var, 0.99)
    else:
        optimizer = tf.compat.v1.train.AdamOptimizer(lr_var)
    step = optimizer.minimize(loss)

    config = tf.compat.v1.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")  # 混合精度训练
    custom_op.parameter_map["use_off_line"].b = True        # 使用NPU训练
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
    
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        model_vars = adda.util.collect_vars(model)
        saver = tf.compat.v1.train.Saver(var_list=model_vars)
        output_dir = os.path.join('snapshot', output)
        if not os.path.exists('snapshot'):
            os.mkdir('snapshot')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        losses = deque(maxlen=10)
        best_acc = 0
        i = 0
        total_time = 0
        while True:
            try:
                start = time.time()
                im_batch_,label_batch_ = sess.run([im_batch,label_batch])
                loss_val, _ = sess.run([loss, step],feed_dict={im_batch_p: im_batch_, label_batch_p: label_batch_})
                end = time.time()
                losses.append(loss_val)
                i = i + 1
                total_time += end-start 
                if i % display == 0:
                    logging.info('{:20} {:10.4f}     (avg: {:10.4f})'.format('Iteration {}:'.format(i),loss_val,np.mean(losses)))
                    if i % 200 == 0:
                        print('*******total_time********', total_time)
                        total_time = 0
                    if dataset_name == 'svhn':
                        dataset = MNIST()
                        im, label = dataset.train.images, dataset.train.labels
                    elif dataset_name == 'usps1800':
                        dataset = MNIST2000()
                        im, label = dataset.train.images, dataset.train.labels
                    elif dataset_name == 'mnist2000':
                        dataset = USPS1800()
                        im, label = dataset.train.images, dataset.train.labels
                        im = adda.models.preprocessing(im, model_fn)
                        im = im.eval()
                    acc = accuracy.eval(feed_dict={im_batch_p: im, label_batch_p: label})
                    print('----------acc------------', acc)
                if (i + 1) >= snapshot and acc > best_acc:
                    best_acc = acc
                    snapshot_path = saver.save(sess, os.path.join(output_dir, output))
                    logging.info('Saved snapshot to {}'.format(snapshot_path))
                if i > iterations:
                    break
                    
            except tf.errors.OutOfRangeError:
                print('best_acc:', best_acc)
                print("iterator done")
                break

            
if __name__ == '__main__':
    main()