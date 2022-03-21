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
import sys
import random
from collections import deque
import click
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tqdm import tqdm
import time
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import adda
from adda.data.mnist import MNIST,MNIST2000
from adda.data.usps import USPS, USPS1800
from adda.data.svhn import SVHN
import npu_bridge
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import moxing as mox


@click.command()
@click.argument('source')
@click.argument('target')
@click.argument('model')
@click.argument('output')
@click.option('--gpu', default='0')
@click.option('--iterations', default=20000)
@click.option('--batch_size', default=50)
@click.option('--display', default=10)
@click.option('--lr', default=1e-4)
@click.option('--stepsize', type=int)
@click.option('--snapshot', default=5000)
@click.option('--weights', required=True)
@click.option('--solver', default='sgd')
@click.option('--adversary', 'adversary_layers', default=[500, 500],
              multiple=True)
@click.option('--adversary_leaky/--adversary_relu', default=True)
@click.option('--seed',default=0, type=int)
def main(source, target, model, output,
         gpu, iterations, batch_size, display, lr, stepsize, snapshot, weights,
         solver, adversary_layers, adversary_leaky, seed):
    adda.util.config_logging()
    logging.info('Using random seed {}'.format(seed))
    random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)
    np.random.seed(0)
    tf.set_random_seed(0)
    error = False
    try:
        source_dataset_name, source_split_name = source.split(':')
    except ValueError:
        error = True
        logging.error(
            'Unexpected source dataset {} (should be in format dataset:split)'
            .format(source))
    try:
        target_dataset_name, target_split_name = target.split(':')
    except ValueError:
        error = True
        logging.error(
            'Unexpected target dataset {} (should be in format dataset:split)'
            .format(target))
    if error:
        raise click.Abort
        
    # setup data
    logging.info('Adapting {} -> {}'.format(source, target))
    source_dataset = getattr(adda.data.get_dataset(source_dataset_name),
                             source_split_name)
    target_dataset = getattr(adda.data.get_dataset(target_dataset_name),
                             target_split_name)
    source_im_batch, source_label_batch = source_dataset.tf_ops_data(iterations,batch_size)
    target_im_batch, target_label_batch = target_dataset.tf_ops_data(iterations,batch_size)

    source_im_batch_p = tf.placeholder(tf.float32, [None,28,28,1])
    source_label_batch_p = tf.placeholder(tf.int32, [None])
    target_im_batch_p = tf.placeholder(tf.float32, [None,28,28,1])
    target_label_batch_p = tf.placeholder(tf.int32, [None])
    
    model_fn = adda.models.get_model_fn(model)

    # base network
    source_ft, _ = model_fn(source_im_batch_p, scope='source')
    target_ft, _ = model_fn(target_im_batch_p, scope='target')
    
    # accuracy
    correct_prediction = tf.equal(tf.argmax(target_ft,1), tf.cast(target_label_batch_p,tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # adversarial network
    source_ft = tf.reshape(source_ft, [-1, int(source_ft.get_shape()[-1])])
    target_ft = tf.reshape(target_ft, [-1, int(target_ft.get_shape()[-1])])
    adversary_ft = tf.concat([source_ft, target_ft], 0)
    source_adversary_label = tf.zeros([tf.shape(source_ft)[0]], tf.int32)
    target_adversary_label = tf.ones([tf.shape(target_ft)[0]], tf.int32)
    adversary_label = tf.concat(
        [source_adversary_label, target_adversary_label], 0)
    adversary_logits = adda.adversary.adversarial_discriminator(
        adversary_ft, adversary_layers, leaky=adversary_leaky)

    # losses
    mapping_loss = tf.losses.sparse_softmax_cross_entropy(
        1 - adversary_label, adversary_logits)
    adversary_loss = tf.losses.sparse_softmax_cross_entropy(
        adversary_label, adversary_logits)

    # variable collection
    source_vars = adda.util.collect_vars('source')
    target_vars = adda.util.collect_vars('target')
    adversary_vars = adda.util.collect_vars('adversary')

    # optimizer
    lr_var = tf.Variable(lr, name='learning_rate', trainable=False)
    if solver == 'sgd':
        optimizer = tf.train.MomentumOptimizer(lr_var, 0.99)
    else:
        optimizer = tf.train.AdamOptimizer(lr_var, 0.5)
    mapping_step = optimizer.minimize(mapping_loss, var_list=list(target_vars.values()))
    adversary_step = optimizer.minimize(adversary_loss, var_list=list(adversary_vars.values()))
    
    config = tf.compat.v1.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")  # 混合精度训练
    custom_op.parameter_map["use_off_line"].b = True
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
    
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
         # restore weights
        if os.path.isdir(weights):
            weights = tf.train.latest_checkpoint(weights)
        logging.info('Restoring weights from {}:'.format(weights))
        logging.info('    Restoring source model:')
        for src, tgt in source_vars.items():
            logging.info('        {:30} -> {:30}'.format(src, tgt.name))
        source_restorer = tf.train.Saver(var_list=source_vars)
        source_restorer.restore(sess, weights)
        logging.info('    Restoring target model:')
        for src, tgt in target_vars.items():
            logging.info('        {:30} -> {:30}'.format(src, tgt.name))
        target_restorer = tf.train.Saver(var_list=target_vars)
        target_restorer.restore(sess, weights)

        # optimization loop (finally)
        output_dir = os.path.join('snapshot', output)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        mapping_losses = deque(maxlen=10)
        adversary_losses = deque(maxlen=10)
        best_acc = 0
        i = 0
        total_time = 0
        while True:
            try:
                start = time.time()
                source_im_batch_,source_label_batch_,target_im_batch_,target_label_batch_ = sess.run([source_im_batch,source_label_batch,target_im_batch,target_label_batch])
                mapping_loss_val, adversary_loss_val, _, _ = sess.run([mapping_loss, adversary_loss, mapping_step, adversary_step],
                                                                      feed_dict={source_im_batch_p:source_im_batch_
                                                                          ,source_label_batch_p:source_label_batch_
                                                                      ,target_im_batch_p:target_im_batch_
                                                                      ,target_label_batch_p:target_label_batch_})
                end = time.time()
                mapping_losses.append(mapping_loss_val)
                adversary_losses.append(adversary_loss_val)
                i = i + 1
                total_time += end-start
                if i % display == 0:
                    logging.info('{:20} Mapping: {:10.4f}     (avg: {:10.4f})'
                                '    Adversary: {:10.4f}     (avg: {:10.4f})'
                                .format('Iteration {}:'.format(i),
                                        mapping_loss_val,
                                        np.mean(mapping_losses),
                                        adversary_loss_val,
                                        np.mean(adversary_losses)))
                    if target_dataset_name == 'mnist':
                        dataset = MNIST().train
                        im, label = dataset.images, dataset.labels
                    elif target_dataset_name == 'mnist2000':
                        dataset = MNIST2000().train
                        im, label = dataset.images, dataset.labels
                    elif target_dataset_name == 'usps1800':
                        dataset = USPS1800().train
                        im, label = dataset.images, dataset.labels
                        im = adda.models.preprocessing(im,model_fn)
                        im = im.eval()
                    acc = sess.run(accuracy, feed_dict={target_im_batch_p: im,
                                                        target_label_batch_p: label})
                    print('----------acc------------', acc)
                    if i % 200 == 0:
                        print('*******total_time********', total_time)
                        total_time = 0
                if (i + 1) >= snapshot and acc > best_acc:
                    best_acc = acc
                    print('best_acc:',best_acc)
                    snapshot_path = target_restorer.save(sess, os.path.join(output_dir, output))
                    logging.info('Saved snapshot to {}'.format(snapshot_path))
            except tf.errors.OutOfRangeError:
                print('best_acc:', best_acc)
                print("iterator done")
                # tf.io.write_graph(sess.graph_def, os.path.join('/home/ma-user/modelarts/workspace/device0/', output), 'graph_adda.pbtxt')
                break

if __name__ == '__main__':
    main()