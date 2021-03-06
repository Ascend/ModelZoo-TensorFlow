# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import sys
import time
import json
import argparse

import numpy as np
import tensorflow as tf
from tensorflow_core.python.tools import freeze_graph

import nn as real_nvp_nn
from model import model_spec as real_nvp_model_spec
from model import inv_model_spec as real_nvp_inv_model_spec
import cifar10_data as cifar10_data
import util
import plotting


from npu_bridge.estimator import npu_ops 
from npu_bridge.estimator.npu_unary_ops import npu_unary_ops
from npu_bridge.estimator.npu.npu_config import NPURunConfig
from npu_bridge.estimator.npu.npu_estimator import NPUEstimator,NPUEstimatorSpec
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
from npu_bridge.estimator.npu import npu_loss_scale_optimizer
from npu_bridge.estimator.npu import npu_loss_scale_manager
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
gpu_thread_count = 2
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = str(gpu_thread_count)
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-z', '--num_gpus', type=int, default=1, help='Location for the dataset')  #for huawei, compat with nr_gpu

parser.add_argument('-i', '--data_url', type=str, default='cifar-10-python/cifar-10-batches-py/', help='Location for the dataset')
parser.add_argument('-o', '--train_url', type=str, default='out/', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=2, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=int, default=0, help='Restore training from previous model checkpoint? 1 = Yes, 0 = No')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=12, help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=1, help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--sample_batch_size', type=int, default=16, help='How many images to process in paralell during sampling?')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)



# Used for training work, but no output!


# initialize data loaders for train/test splits
DataLoader = cifar10_data.DataLoader
train_data = DataLoader('/data1/NRE_Check/realnvp/RealNVP_tf_hw10669533/cifar-10-python/cifar-10-batches-py', 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True)
test_data = DataLoader('/data1/NRE_Check/realnvp/RealNVP_tf_hw10669533/cifar-10-python/cifar-10-batches-py', 'test', args.batch_size * args.nr_gpu, shuffle=False)

obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)
model_spec = real_nvp_model_spec
inv_model_spec = real_nvp_inv_model_spec
nn = real_nvp_nn

# create the model
model = tf.make_template('model', model_spec)
inv_model = tf.make_template('model', inv_model_spec, unique_name_='model')

x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
# run once for data dependent initialization of parameters
gen_par = model(x_init)

# sample from the model
x_sample = tf.placeholder(tf.float32, shape=(args.sample_batch_size, ) + obs_shape)
new_x_gen = inv_model(x_sample)
def sample_from_model(sess):
  x_gen = np.random.normal(0.0, 1.0, (args.sample_batch_size,) + obs_shape)
  new_x_gen_np = sess.run(new_x_gen, {x_sample: x_gen})
  return new_x_gen_np

# get loss gradients over multiple GPUs
xs = []
grads = []
loss_gen = []
loss_gen_test = []
all_params = tf.trainable_variables()
######
# for i in range(args.nr_gpu):
xs.append(tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape))
# with tf.device('/gpu:%d' % i):
# train
gen_par,jacs = model(xs[0])
loss_gen.append(nn.loss(gen_par, jacs))
# gradients
grads.append(tf.gradients(loss_gen[0], all_params))
# test
gen_par2,jacs2 = model(xs[0])
gen_par2 = tf.add(gen_par2, 0, name="gen")
jacs2 = tf.add(jacs2, 0, name='jac')
# gen_par2=tf.cast(gen_par2,gen_par2.dtype,name="par")
# jacs2=tf.cast(jacs2,jacs2.dtype,name="jac")
loss_gen_test.append(nn.loss(gen_par2, jacs2))


# add gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
optimizer =tf.train.AdamOptimizer(learning_rate=tf_lr)
optimizer=optimizer.minimize(loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size))
######
# convert loss to bits/dim
bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
bits_per_dim_test = loss_gen_test[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)

# init & save
initializer = tf.initialize_all_variables()
saver = tf.train.Saver()

# input is scaled from uint8 [0,255] to float in range [-1,1]
def prepro(x):
  return np.cast[np.float32]((x - 127.5) / 127.5)

def compute_likelihood(xf):
  print ("computing likelihood of image with mean %f" % np.mean(xf))
  xfs = np.split(xf, args.nr_gpu)
  feed_dict = { xs[i]: xfs[i] for i in range(args.nr_gpu) }
  l = sess.run(bits_per_dim_test, feed_dict)
  return l

# //////////// perform training //////////////
if not os.path.exists(args.train_url):
    os.makedirs(args.train_url)
print('starting training')
test_bpd = []
lr = args.learning_rate

config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_fp32_to_fp16")
# custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("must_keep_origin_dtype")
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
# config.graph_options.rewrite_options.optimizers.extend(["GradFusionOptimizer"])
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # level '1':all '2':w+e '3':e


def list_all_tensors_in_pb(model_dir,model_name):
    def create_graph():
        print(os.listdir(model_dir))
        print(os.path.join(model_dir, model_name))
        with tf.gfile.FastGFile(os.path.join(
                model_dir, model_name), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    create_graph()
    tensor_name_list = []
    for tensor in sess.graph_def.node:
        if('par' in tensor.name):
            print(tensor.name)
        elif( 'Adam' in tensor.name):
            pass
        else:
            tensor_name_list.append(tensor.name)
    print('list')
    for tensor_name in tensor_name_list:
        print(tensor_name, '\n')


with tf.Session(config=config) as sess:
    for epoch in range(args.max_epochs):
        begin = time.time()

        # init
        if epoch == 0:
            x = train_data.next(args.init_batch_size) # manually retrieve exactly init_batch_size examples
            train_data.reset() # rewind the iterator back to 0 to do one full epoch
            print('initializing the model...')
            sess.run(initializer,{x_init: prepro(x)})
            if args.load_params:
                ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
                print('restoring parameters from', ckpt_file)
                saver.restore(sess, ckpt_file)
            # util.show_all_variables() # code for debug

        # train for one epoch
        train_losses = []
        print ("Training (%d/%d) started" % (epoch, args.max_epochs))
        for t,x in enumerate(train_data):
          # prepro the data and split it for each gpu
          xf = prepro(x)
          xfs = np.split(xf, args.nr_gpu)
          lr *= args.lr_decay
          feed_dict = { tf_lr: lr }
          feed_dict.update({ xs[i]: xfs[i] for i in range(args.nr_gpu) })


          start_time = time.time()
          l,_ = sess.run([bits_per_dim, optimizer], feed_dict) # toooooo long
          end_time = time.time()-start_time
          print('--train_time--',end_time)
          
          # l = sess.run(bits_per_dim, feed_dict) # if debug, use this func to speed up (useless for result)
          train_losses.append(l)
        train_loss_gen = np.mean(train_losses)

        # compute likelihood over test split
        test_losses = []
        print ("Testing...")
        for x in test_data:
          xf = prepro(x)
          xfs = np.split(xf, args.nr_gpu)
          feed_dict = { xs[i]: xfs[i] for i in range(args.nr_gpu) }
          l = sess.run(bits_per_dim_test, feed_dict)

          test_losses.append(l)
        test_loss_gen = np.mean(test_losses)
        test_bpd.append(test_loss_gen)

        # log progress to console
        print("Iteration %d, time = %d, train bits_per_dim = %.4f, test bits_per_dim = %.4f" % (epoch, time.time()-begin, train_loss_gen, test_loss_gen))
        sys.stdout.flush()

        if epoch % args.save_interval == 0:

            print ("Generating samples...")

            # save params
            cwd = os.getcwd()  # current path
            t=str(time.time())
            #os.makedirs("save_model_"+t,exist_ok=True)
            #os.makedirs("save_testbpd_"+t,exist_ok=True)
            #os.makedirs("pb_model_"+t,exist_ok=True)

            save_path = 'save_model_'+t+'/params_cifar.ckpt'
            osr_dir=os.path.join(cwd, 'save_model_'+t)
            osr=os.path.join(cwd, save_path)

            ret=saver.save(sess, osr) # main func  # output /home/work/workspace/device7//data1/NRE_Check/realnvp/RealNVP_tf_hw10669533/test/..//test/output/3/ckpt12345/params_cifar.ckpt

            