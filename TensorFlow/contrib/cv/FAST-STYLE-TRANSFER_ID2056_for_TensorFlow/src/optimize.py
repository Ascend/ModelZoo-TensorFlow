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

from __future__ import print_function
import functools
import pdb, time
import src.vgg as vgg
import tensorflow as tf, numpy as np, os
import src.transform as transform
from src.utils import get_img
from cfg import make_config

from npu_bridge.npu_init import *
# 改
# from npu_bridge.hccl import hccl_ops
#import precision_tool.tf_config as npu_tf_config

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'
#DEVICES = 'CUDA_VISIBLE_DEVICES'
# 改
rank_size = int(os.getenv('RANK_SIZE'))
rank_id = int(os.getenv('RANK_ID'))
print(rank_id, rank_size)
# np arr, np arr
def optimize(content_targets, style_target, content_weight, style_weight,
             tv_weight, vgg_path, epochs=2, print_iterations=1000,
             batch_size=4, save_path='saver/fns.ckpt', slow=False,
             learning_rate=1e-3, debug=False):
    if slow:
        batch_size = 1
    # 改
    mod = len(content_targets) % (batch_size*rank_size)
    if mod > 0:
        print("Train set has been trimmed slightly..")
        content_targets = content_targets[:-mod]

    style_features = {}

    batch_shape = (batch_size,256,256,3)
    style_shape = (1,) + style_target.shape
    print(style_shape)
    #soft_config = make_config('npu')
    # precompute style features
    with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
        style_image = tf.compat.v1.placeholder(tf.float32, shape=style_shape, name='style_image')
        style_image_pre = vgg.preprocess(style_image)
        net = vgg.net(vgg_path, style_image_pre)
        style_pre = np.array([style_target])
        for layer in STYLE_LAYERS:
            features = net[layer].eval(feed_dict={style_image:style_pre})
            features = np.reshape(features, (-1, features.shape[3]))
            gram = np.matmul(features.T, features) / features.size
            style_features[layer] = gram

    soft_config = make_config('npu')
    with tf.Graph().as_default(), tf.compat.v1.Session(config=soft_config) as sess:
        X_content = tf.compat.v1.placeholder(tf.float32, shape=batch_shape, name="X_content")
        X_pre = vgg.preprocess(X_content)

        # precompute content features
        content_features = {}
        content_net = vgg.net(vgg_path, X_pre)
        content_features[CONTENT_LAYER] = content_net[CONTENT_LAYER]

        if slow:
            preds = tf.Variable(
                tf.random.normal(X_content.get_shape()) * 0.256
            )
            preds_pre = preds
        else:
            preds = transform.net(X_content/255.0)
            preds_pre = vgg.preprocess(preds)

        net = vgg.net(vgg_path, preds_pre)

        content_size = _tensor_size(content_features[CONTENT_LAYER])*batch_size
        assert _tensor_size(content_features[CONTENT_LAYER]) == _tensor_size(net[CONTENT_LAYER])
        content_size = tf.cast(content_size, tf.float32)
        content_loss = content_weight * (2 * tf.nn.l2_loss(
            net[CONTENT_LAYER] - content_features[CONTENT_LAYER]) / content_size
        )

        style_losses = []
        #lsum = 0
        for style_layer in STYLE_LAYERS:
            layer = net[style_layer]
            bs, height, width, filters = map(lambda i:i,layer.get_shape())
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(a=feats, perm=[0,2,1])
            size = tf.cast(size, tf.float32)
            grams = tf.matmul(feats_T, feats) / size
            style_gram = style_features[style_layer]
            style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)
            #lsum = lsum + 2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size
            #print(style_losses)
        style_loss = style_weight * functools.reduce(tf.add, style_losses) / batch_size
        #style_loss = style_weight * lsum / batch_size

        # total variation denoising
        tv_y_size = _tensor_size(preds[:,1:,:,:])
        tv_x_size = _tensor_size(preds[:,:,1:,:])
        y_tv = tf.nn.l2_loss(preds[:,1:,:,:] - preds[:,:batch_shape[1]-1,:,:])
        x_tv = tf.nn.l2_loss(preds[:,:,1:,:] - preds[:,:,:batch_shape[2]-1,:])
        ##batch_size = tf.cast(batch_size, tf.float32)
        tv_x_size = tf.cast(tv_x_size, tf.float32)
        tv_y_size = tf.cast(tv_y_size, tf.float32)
        tv_loss = tv_weight*2*(x_tv/tv_x_size + y_tv/tv_y_size)/batch_size

        loss = content_loss + style_loss + tv_loss
        #print(loss)
        # overall loss

        # change
        if int(rank_size) > 1:
            train_step = npu_distributed_optimizer_wrapper(tf.compat.v1.train.AdamOptimizer(learning_rate)).minimize(loss)
        else :
            train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

        '''
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate)
        loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        loss_scale_optimizer = NPULossScaleOptimizer(opt, loss_scale_manager)
        train_step = loss_scale_optimizer.minimize(loss)
        '''
        sess.run(tf.compat.v1.global_variables_initializer())
        # 改 广播
        if int(rank_size) > 1:
            input = tf.trainable_variables()
            bcast_global_variables_op = hccl_ops.broadcast(input, 0)
            # custom_op.parameter_map["hcom_parallel"].b = True
            sess.run(bcast_global_variables_op)
        #train_step = util.set_iteration_per_loop(sess, train_step, 10)
        import random
        uid = random.randint(1, 100)
        print("UID: %s" % uid)
        for epoch in range(epochs):
            num_examples = len(content_targets)
            iterations = 0
            #batch_size = int(batch_size)
            #print('1')
            #print(type(batch_size))
            #print(type(iterations))
            #print(type(num_examples))

            #print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')
            # 改
            while iterations * batch_size * rank_size < num_examples:

                start_time = time.time()
                # 改
                curr = iterations * batch_size*rank_size+batch_size*rank_id
                step = curr + batch_size
                X_batch = np.zeros(batch_shape, dtype=np.float32)
                for j, img_p in enumerate(content_targets[curr:step]):
                   X_batch[j] = get_img(img_p, (256,256,3)).astype(np.float32)

                #print('bbbbbbbbbbbbbbbbbbbb')

                iterations += 1
                assert X_batch.shape[0] == batch_size

                feed_dict = {
                   X_content:X_batch
                }

                train_step.run(feed_dict=feed_dict)
                end_time = time.time()
                delta_time = end_time - start_time
                print("UID: %s, batch time: %s" % (uid, delta_time))
                # 改
                is_print_iter = int(iterations) % print_iterations == 0
                if slow:
                    is_print_iter = epoch % print_iterations == 0
                # 改
                is_last = epoch == epochs - 1 and iterations * batch_size*rank_size >= num_examples
                should_print = is_print_iter or is_last
                if should_print:
                    to_get = [style_loss, content_loss, tv_loss, loss, preds]
                    test_feed_dict = {
                       X_content:X_batch
                    }

                    tup = sess.run(to_get, feed_dict = test_feed_dict)
                    _style_loss,_content_loss,_tv_loss,_loss,_preds = tup

                    # losses = (_style_loss, _content_loss, _tv_loss, _loss)
                    if slow:
                       _preds = vgg.unprocess(_preds)
                    else:
                       saver = tf.compat.v1.train.Saver()
                       res = saver.save(sess, save_path)
                    #yield(_preds, losses, iterations, epoch)
                    # style_loss, content_loss, tv_loss, loss = losses

                    print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, int(iterations), _loss))
                    to_print = (_style_loss, _content_loss, _tv_loss)
                    print('style: %s, content:%s, tv: %s' % to_print)
					
					

def _tensor_size(tensor):
    from operator import mul
    return functools.reduce(mul, (d for d in tensor.get_shape()[1:]), 1)
