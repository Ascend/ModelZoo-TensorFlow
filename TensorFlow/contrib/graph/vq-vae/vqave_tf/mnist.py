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
# Copyright 2020 Huawei Technologies Co., Ltd
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
from six.moves import xrange
import os
import better_exceptions
import tensorflow as tf
import time
import numpy as np
from tqdm import tqdm

from npu_bridge.estimator.npu import npu_scope
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import argparse

from model import VQVAE, _mnist_arch, PixelCNN

tf.app.flags.DEFINE_string('data_path', "datasets/mnist", '')
tf.app.flags.DEFINE_integer('TRAIN_NUM', 20000, '')
FLAGS = tf.app.flags.FLAGS

def main(config,
         RANDOM_SEED,
         LOG_DIR,
         TRAIN_NUM,
         BATCH_SIZE,
         LEARNING_RATE,
         DECAY_VAL,
         DECAY_STEPS,
         DECAY_STAIRCASE,
         BETA,
         K,
         D,
         SAVE_PERIOD,
         SUMMARY_PERIOD,
         **kwargs):
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    # >>>>>>> DATASET
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(FLAGS.data_path, one_hot=False)
    # <<<<<<<

    # >>>>>>> MODEL
    x = tf.placeholder(tf.float32,[None,784])
    resized = tf.image.resize_images(
        tf.reshape(x,[-1,28,28,1]),
        (24,24),
        method=tf.image.ResizeMethod.BILINEAR)

    with tf.variable_scope('train'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_VAL, staircase=DECAY_STAIRCASE)
        tf.summary.scalar('lr',learning_rate)

        with tf.variable_scope('params') as params:
            pass
        net = VQVAE(learning_rate,global_step,BETA,resized,K,D,_mnist_arch,params,True)

    with tf.variable_scope('valid'):
        params.reuse_variables()
        valid_net = VQVAE(None,None,BETA,resized,K,D,_mnist_arch,params,False)

    with tf.variable_scope('misc'):
        # Summary Operations
        tf.summary.scalar('loss',net.loss)
        tf.summary.scalar('recon',net.recon)
        tf.summary.scalar('vq',net.vq)
        tf.summary.scalar('commit',BETA*net.commit)
        tf.summary.image('origin',resized,max_outputs=4)
        tf.summary.image('recon',net.p_x_z,max_outputs=4)
        # TODO: logliklihood

        summary_op = tf.summary.merge_all()

        # Initialize op
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(config.as_matrix()), collections=[])

        extended_summary_op = tf.summary.merge([
            tf.summary.scalar('valid_loss',valid_net.loss),
            tf.summary.scalar('valid_recon',valid_net.recon),
            tf.summary.scalar('valid_vq',valid_net.vq),
            tf.summary.scalar('valid_commit',BETA*valid_net.commit),
            tf.summary.image('valid_recon',valid_net.p_x_z,max_outputs=10),
        ])

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True 
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)

    summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
    summary_writer.add_summary(config_summary.eval(session=sess))

    for step in tqdm(xrange(TRAIN_NUM),dynamic_ncols=True):
        batch_xs, _= mnist.train.next_batch(BATCH_SIZE)
        it,loss,_ = sess.run([global_step,net.loss,net.train_op],feed_dict={x:batch_xs})

        if( it % SAVE_PERIOD == 0 ):
            net.save(sess,LOG_DIR,step=it)

        if( it % SUMMARY_PERIOD == 0 ):
            tqdm.write('[ %5d] Loss: %1.3f'%(it,loss))
            summary = sess.run(summary_op,feed_dict={x:batch_xs})
            summary_writer.add_summary(summary,it)

        if( it % (SUMMARY_PERIOD*2) == 0 ): #Extended Summary
            batch_xs, _= mnist.test.next_batch(BATCH_SIZE)
            summary = sess.run(extended_summary_op,feed_dict={x:batch_xs})
            summary_writer.add_summary(summary,it)



    net.save(sess,LOG_DIR)

def extract_z(MODEL,
              BATCH_SIZE,
              BETA,
              K,
              D,
              **kwargs):
    # >>>>>>> DATASET
    tf.reset_default_graph()
    
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(FLAGS.data_path, one_hot=False)
    # <<<<<<<

    # >>>>>>> MODEL
    x = tf.placeholder(tf.float32,[None,784])
    resized = tf.image.resize_images(
        tf.reshape(x,[-1,28,28,1]),
        (24,24),
        method=tf.image.ResizeMethod.BILINEAR)

    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        net = VQVAE(None,None,BETA,resized,K,D,_mnist_arch,params,False)

    # Initialize op
    init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True 
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    custom_op.parameter_map["dynamic_input"].b = True
    custom_op.parameter_map["dynamic_graph_execute_mode"].s = tf.compat.as_bytes("lazy_recompile")

    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)
    net.load(sess,MODEL)

    xs,ys = mnist.train.images, mnist.train.labels
    ks = []
    for i in tqdm(range(0,len(xs),BATCH_SIZE)):
        batch = xs[i:i+BATCH_SIZE]

        k = sess.run(net.k,feed_dict={x:batch})
        ks.append(k)
    ks = np.concatenate(ks,axis=0)

    np.savez(os.path.join(os.path.dirname(MODEL),'ks_ys.npz'),ks=ks,ys=ys)

def train_prior(config,
                RANDOM_SEED,
                MODEL,
                TRAIN_NUM,
                BATCH_SIZE,
                LEARNING_RATE,
                DECAY_VAL,
                DECAY_STEPS,
                DECAY_STAIRCASE,
                GRAD_CLIP,
                K,
                D,
                BETA,
                NUM_LAYERS,
                NUM_FEATURE_MAPS,
                SUMMARY_PERIOD,
                SAVE_PERIOD,
                **kwargs):
    tf.reset_default_graph()
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)
    LOG_DIR = os.path.join(os.path.dirname(MODEL),'pixelcnn')
    # >>>>>>> DATASET
    class Latents():
        def __init__(self,path,validation_size=5000):
            from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
            from tensorflow.contrib.learn.python.learn.datasets import base

            data = np.load(path)
            train = DataSet(data['ks'][validation_size:], data['ys'][validation_size:],reshape=False,dtype=np.uint8,one_hot=False) #dtype won't bother even in the case when latent is int32 type.
            validation = DataSet(data['ks'][:validation_size], data['ys'][:validation_size],reshape=False,dtype=np.uint8,one_hot=False)
            #test = DataSet(data['test_x'],np.argmax(data['test_y'],axis=1),reshape=False,dtype=np.float32,one_hot=False)
            self.size = data['ks'].shape[1]
            self.data = base.Datasets(train=train, validation=validation, test=None)
    latent = Latents(os.path.join(os.path.dirname(MODEL),'ks_ys.npz'))
    # <<<<<<<

    # >>>>>>> MODEL for Generate Images
    with tf.variable_scope('net'):
        with tf.variable_scope('params') as params:
            pass
        _not_used = tf.placeholder(tf.float32,[None,24,24,1])
        vq_net = VQVAE(None,None,BETA,_not_used,K,D,_mnist_arch,params,False)
    # <<<<<<<

    # >>>>>> MODEL for Training Prior
    with tf.variable_scope('pixelcnn'):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEPS, DECAY_VAL, staircase=DECAY_STAIRCASE)
        tf.summary.scalar('lr',learning_rate)

        net = PixelCNN(learning_rate,global_step,GRAD_CLIP,
                       latent.size,vq_net.embeds,K,D,
                       10,NUM_LAYERS,NUM_FEATURE_MAPS)
    # <<<<<<
    with tf.variable_scope('misc'):
        # Summary Operations
        tf.summary.scalar('loss',net.loss)
        summary_op = tf.summary.merge_all()

        # Initialize op
        init_op = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        config_summary = tf.summary.text('TrainConfig', tf.convert_to_tensor(config.as_matrix()), collections=[])

        sample_images = tf.placeholder(tf.float32,[None,24,24,1])
        sample_summary_op = tf.summary.image('samples',sample_images,max_outputs=20)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Run!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True 
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess = tf.Session(config=config)
    sess.graph.finalize()
    sess.run(init_op)
    vq_net.load(sess,MODEL)

    summary_writer = tf.summary.FileWriter(LOG_DIR,sess.graph)
    summary_writer.add_summary(config_summary.eval(session=sess))

    for step in tqdm(xrange(TRAIN_NUM),dynamic_ncols=True):
        batch_xs, batch_ys = latent.data.train.next_batch(BATCH_SIZE)
        it,loss,_ = sess.run([global_step,net.loss,net.train_op],feed_dict={net.X:batch_xs,net.h:batch_ys})

        if( it % SAVE_PERIOD == 0 ):
            net.save(sess,LOG_DIR,step=it)

        if( it % SUMMARY_PERIOD == 0 ):
            tqdm.write('[ %5d] Loss: %1.3f'%(it,loss))
            summary = sess.run(summary_op,feed_dict={net.X:batch_xs,net.h:batch_ys})
            summary_writer.add_summary(summary,it)

        if( it % (SUMMARY_PERIOD * 2) == 0 ):
            sampled_zs,log_probs = net.sample_from_prior(sess,np.arange(10),2)
            sampled_ims = sess.run(vq_net.gen,feed_dict={vq_net.latent:sampled_zs})
            summary_writer.add_summary(
                sess.run(sample_summary_op,feed_dict={sample_images:sampled_ims}),it)
    print("Final Train Accuracy", loss)

    net.save(sess,LOG_DIR)


def get_default_param():
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        'LOG_DIR':'./log/mnist/%s'%(now),
        'MODEL' : './log/mnist/%s/last.ckpt'%(now),

        'TRAIN_NUM' : FLAGS.TRAIN_NUM, #60000, # Size corresponds to one epoch
        'BATCH_SIZE': 32, #32,

        'LEARNING_RATE' : 0.0002,
        'DECAY_VAL' : 1.0,
        'DECAY_STEPS' : 20000, # Half of the training procedure.
        'DECAY_STAIRCASE' : False,

        'BETA':0.25,
        'K':10, # 5,
        'D':64,

        # PixelCNN Params
        'GRAD_CLIP' : 1.0,
        'NUM_LAYERS' : 12,
        'NUM_FEATURE_MAPS' : 32,

        'SUMMARY_PERIOD' : 100,
        'SAVE_PERIOD' : 10000,
        'RANDOM_SEED': 0,
    }

if __name__ == "__main__":
    class MyConfig(dict):
        pass
    params = get_default_param()
    config = MyConfig(params)
    def as_matrix() :
        return [[k, str(w)] for k, w in config.items()]
    config.as_matrix = as_matrix
    
    start = time.time()

    main(config=config,**config)
    extract_z(**config)
    train_prior(config=config,**config)


    E2Etime = time.time() - start
    print("E2E Training Duration sec", E2Etime)
    print("avg time per step", E2Etime/FLAGS.TRAIN_NUM)
    print("FPS", 32*FLAGS.TRAIN_NUM/E2Etime)

