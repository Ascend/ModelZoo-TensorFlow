# -*- coding: utf-8 -*-
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

#from keras.layers import Embedding
from datetime import datetime
import os.path
import os
import moxing as mox
print(os.listdir(os.getcwd()))
import time
import sys
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,CURRENT_PATH+'/../lib')
sys.path.insert(0,CURRENT_PATH+'/../networks')
from lib import utils
import math
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1, resnet_v2
from networks import MobileFaceNet as mobilenet
from tensorflow import data as tf_data
from collections import Counter
import numpy as np
from scipy import misc
import importlib
import itertools
import argparse
import pdb


trip_thresh = 0

from tensorflow.python.ops import data_flow_ops

def _from_tensor_slices(tensors_x,tensors_y):
    #return TensorSliceDataset((tensors_x,tensors_y))
    return tf_data.Dataset.from_tensor_slices((tensors_x,tensors_y))


def main(args):


    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    np.random.seed(seed=args.seed)
    #！
    #args.num_gpus = 1
    args.dataset = 'webface'
    args.network = 'mobilenet'
    args.data_dir = '/home/ma-user/modelarts/user-job-dir/MassFac/dataset/casia-112x112'
    args.save_dir = '/home/ma-user/modelarts/user-job-dir/MassFac/facenet_ms_mp/'
    #print('datadir')
    #print(args.data_dir)
    print('load data...')
    if args.dataset == 'webface':
        train_set = utils.get_dataset(args.data_dir)
    elif args.dataset == 'mega':
        train_set = utils.dataset_from_cache(args.data_dir)
    #train_set.extend(ic_train_set)
    print('Loaded dataset: {} persons'.format(len(train_set)))
    nrof_classes = len(train_set)
    class_indices = list(np.arange(nrof_classes))
    #np.random.shuffle(class_indices)
    #print(class_indices)

    def _sample_people(x):
        '''We sample people based on tf.data, where we can use transform and prefetch.

        '''
        scale = 1 if args.mine_method != 'simi_online' else args.scale 
        image_paths, labels = sample_people(train_set,class_indices,args.people_per_batch*args.num_gpus*scale,args.images_per_person)
        #labels = []
        #print(labels)
        #for i in range(len(num_per_class)):
        #    labels.extend([i]*num_per_class[i])
        return (np.array(image_paths),np.array(labels,dtype=np.int32))

    def _parse_function(filename,label):
        file_contents = tf.read_file(filename)

        image = tf.image.decode_image(file_contents, channels=3)
        #image = tf.image.decode_jpeg(file_contents, channels=3)
        if args.random_crop:
            print('use random crop')
            image = tf.random_crop(image, [args.image_size, args.image_size, 3])
        else:
            print('Not use random crop')
            #image.set_shape((args.image_size, args.image_size, 3))
            image.set_shape((None,None, 3))
            image = tf.image.resize_images(image, size=(args.image_size, args.image_size))

        if args.random_flip:
            image = tf.image.random_flip_left_right(image)

        #pylint: disable=no-member
        image.set_shape((args.image_size, args.image_size, 3))
        print('img shape',image.shape)
        image = tf.cast(image,tf.float32)
        image = tf.subtract(image,127.5)
        image = tf.div(image,128.)
        return image, label

    gpus = range(args.num_gpus)
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False,name='global_step')
        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        #the image is generated by sequence
        single_batch_size = args.people_per_batch*args.images_per_person
        total_batch_size = args.num_gpus*single_batch_size
        with tf.device("/cpu:0"):
            dataset = tf_data.Dataset.range(args.epoch_size*args.max_nrof_epochs*100)
            #dataset.repeat(args.max_nrof_epochs)
            #sample people based map
            dataset = dataset.map(lambda x: tf.py_func(_sample_people,[x],[tf.string,tf.int32]))
            dataset = dataset.flat_map(_from_tensor_slices)
            dataset = dataset.map(_parse_function,num_parallel_calls=8)
            dataset = dataset.batch(total_batch_size,drop_remainder=True)
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()
            batch_image_split = tf.split(next_element[0],args.num_gpus)
            batch_label = next_element[1]
            
            global trip_thresh
            trip_thresh = args.num_gpus*args.people_per_batch*args.images_per_person * 10

        #learning_rate = tf.train.exponential_decay(args.learning_rate, global_step,
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        opt = utils.get_opt(args.optimizer,learning_rate)
       
        tower_embeddings = []
        tower_feats = []
        for i in range(len(gpus)):
            with tf.device("/gpu:" + str(gpus[i])):
                with tf.name_scope("tower_" + str(gpus[i])) as scope:
                  with slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0"):
                    # Build the inference graph
                    with tf.variable_scope(tf.get_variable_scope()) as var_scope:
                        reuse = False if i ==0 else True
                        if args.network == 'resnet_v2': 
                          with slim.arg_scope(resnet_v2.resnet_arg_scope(args.weight_decay)):
                            #prelogits, end_points = resnet_v1.resnet_v1_50(batch_image_split[i], is_training=phase_train_placeholder, output_stride=16, num_classes=args.embedding_size, reuse=reuse)
                            prelogits, end_points = resnet_v2.resnet_v2_50(batch_image_split[i],is_training=True,
                                        output_stride=16,num_classes=args.embedding_size,reuse=reuse)
                            prelogits = tf.squeeze(prelogits, [1,2], name='SpatialSqueeze')
                        elif args.network == 'resface':
                            prelogits, end_points = resface.inference(batch_image_split[i],1.0,bottleneck_layer_size=args.embedding_size,weight_decay=args.weight_decay,reuse=reuse)
                            print('res face prelogits',prelogits)
                        elif args.network ==  'mobilenet':
                            prelogits, net_points = mobilenet.inference(batch_image_split[i],bottleneck_layer_size=args.embedding_size,phase_train=True,weight_decay=args.weight_decay,reuse=reuse)
                        if 1==1:
                            print('use fc bn')
                            embeddings = slim.batch_norm(prelogits, is_training=True, decay=0.997,epsilon=1e-5,scale=True,updates_collections=tf.GraphKeys.UPDATE_OPS,reuse=reuse,scope='softmax_bn')
                            embeddings = tf.nn.l2_normalize(embeddings, 1, 1e-10, name='embeddings')

                        tf.get_variable_scope().reuse_variables()
                    tower_embeddings.append(embeddings)
        embeddings_gather = tf.concat(tower_embeddings,axis=0,name='embeddings_concat')
        if args.with_softmax:
            coco_loss = utils.cos_loss(embeddings_gather,batch_label,len(train_set))
        # select triplet pair by tf op
        with tf.name_scope('triplet_part'):
            #embeddings_norm = tf.nn.l2_normalize(embeddings_gather,axis=1)
            #distances = utils._pairwise_distances(embeddings_norm,squared=True)
            distances = utils._pairwise_distances(embeddings_gather,squared=True)
            #！
            args.strategy = 'min_and_max'

            print('triplet strategy',args.strategy)
            if args.strategy == 'min_and_min':
                pair = tf.py_func(select_triplets_min_min,[distances,batch_label,args.alpha],tf.int64)
            elif args.strategy == 'min_and_max':
                pair = tf.py_func(select_triplets_min_max,[distances,batch_label,args.alpha],tf.int64)
            elif args.strategy == 'hardest':
                pair = tf.py_func(select_triplets_hardest,[distances,batch_label,args.alpha],tf.int64)
            elif args.strategy == 'batch_random': 
                pair = tf.py_func(select_triplets_batch_random,[distances,batch_label,args.alpha],tf.int64)
            elif args.strategy == 'batch_all': 
                pair = tf.py_func(select_triplets_batch_all,[distances,batch_label,args.alpha],tf.int64)
            else:
                raise ValueError('Not supported strategy {}'.format(args.strategy))
            triplet_handle = {}
            triplet_handle['embeddings'] = embeddings_gather
            triplet_handle['labels'] = batch_label
            triplet_handle['pair'] = pair
        #！
        args.mine_method='online'
        if args.mine_method == 'online':
            pair_reshape = tf.reshape(pair,[-1])
            embeddings_gather = tf.gather(embeddings_gather,pair_reshape)
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings_gather, [-1,3,args.embedding_size]), 3, 1)
        triplet_loss, pos_d, neg_d = utils.triplet_loss(anchor, positive, negative, args.alpha)
        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        triplet_loss = tf.add_n([triplet_loss])
        total_loss = triplet_loss + tf.add_n(regularization_losses)
        if args.with_softmax:
            total_loss = total_loss + args.softmax_loss_weight*coco_loss
        #total_loss =  tf.add_n(regularization_losses)
        losses = {}
        losses['triplet_loss'] = triplet_loss
        losses['total_loss'] = total_loss

        update_vars = tf.trainable_variables() 
        with tf.device("/gpu:" + str(gpus[0])):
                    grads = opt.compute_gradients(total_loss,update_vars,colocate_gradients_with_ops=True)

        if args.pretrain_softmax:
            softmax_vars = [var for var in update_vars if 'centers_var' in var.name]
            print('softmax vars',softmax_vars)
            softmax_grads = opt.compute_gradients(coco_loss,softmax_vars)
            softmax_update_op = opt.apply_gradients(softmax_grads)
        apply_gradient_op = opt.apply_gradients(grads,global_step=global_step) 
        #update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'pair_part' in op.name] 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
        print('update ops',update_ops)
        with tf.control_dependencies(update_ops):
            train_op_dep = tf.group(apply_gradient_op)
        train_op = tf.cond(tf.is_nan(triplet_loss), lambda: tf.no_op('no_train'), lambda: train_op_dep)
         
        save_vars = [var  for var in tf.global_variables() if 'Adagrad' not in var.name and 'global_step' not in var.name ]
        restore_vars = [var  for var in tf.global_variables() if 'Adagrad' not in var.name and 'global_step' not in var.name and 'pair_part' not in var.name and 'centers_var' not in var.name]
        print('restore vars',restore_vars)
        saver = tf.train.Saver(save_vars, max_to_keep=3)
        restorer = tf.train.Saver(restore_vars, max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        #!gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        #!sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True))
        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap

        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)
        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(tf.local_variables_initializer(), feed_dict={phase_train_placeholder:True})
        sess.run(iterator.initializer)

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        
        forward_embeddings = []
        with sess.as_default():
            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                #!bug!restorer.restore(sess, os.path.expanduser(args.pretrained_model))
                module_file = tf.train.latest_checkpoint(os.path.expanduser(args.pretrained_model))

                sess.run(tf.global_variables_initializer())
                if module_file is not None:
                    saver.restore(sess, module_file)
            # Training and validation loop
            if args.pretrain_softmax:
                total_images = len(train_set) * 20
                lr_init = 0.1
                for epoch in range(args.softmax_epoch):
                    for i in range(total_images//total_batch_size):
                        if epoch == 4:
                            lr_init = 0.05
                        if epoch == 7:
                            lr_init = 0.01
                        coco_loss_err,_ = sess.run([coco_loss,softmax_update_op],feed_dict={phase_train_placeholder: False,learning_rate_placeholder:lr_init})
                        print('{}/{} {} coco loss err:{} lr:{}'.format(i,total_images//total_batch_size,epoch,coco_loss_err,lr_init))

            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                if args.mine_method == 'simi_online':
                  train_simi_online(args, sess, epoch, len(gpus),embeddings_gather,batch_label,next_element[0],batch_image_split,learning_rate_placeholder,
                     learning_rate, phase_train_placeholder, global_step, pos_d, neg_d, triplet_handle,
                     losses, train_op, summary_op, summary_writer, args.learning_rate_schedule_file)
                elif args.mine_method == 'online':
                  train_online(args, sess, epoch, 
                     learning_rate, phase_train_placeholder, global_step, 
                     losses, train_op, summary_op, summary_writer, args.learning_rate_schedule_file)
                else:
                  raise ValueError('Not supported mini method {}'.format(args.mine_method))
                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)
    return model_dir

def train_simi_online(args, sess, epoch, num_gpus,embeddings_gather,batch_label,images,batch_image_split, learning_rate_placeholder,
          learning_rate, phase_train_placeholder, global_step, pos_d, neg_d,triplet_handle,
          loss, train_op, summary_op, summary_writer, learning_rate_schedule_file):
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = utils.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    batch_number = 0
    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        embeddings_list = []
        labels_list = []
        images_list = []
        f_time = time.time()
        for i in range(args.scale):
            embeddings_np,labels_np,images_np = sess.run([embeddings_gather,batch_label,images],feed_dict={phase_train_placeholder:False,learning_rate_placeholder:lr})
            embeddings_list.append(embeddings_np)
            labels_list.append(labels_np)
            images_list.append(images_np)
        embeddings_all = np.vstack(embeddings_list)
        labels_all = np.hstack(labels_list)
        images_all = np.vstack(images_list)
        print('forward time: {}'.format(time.time()-f_time))
        f_time = time.time()
        triplet_pairs = sess.run(triplet_handle['pair'],feed_dict={triplet_handle['embeddings']: embeddings_all,triplet_handle['labels']: labels_all})
        print('tf op select triplet time: {}'.format(time.time()-f_time))
        triplet_images_size = len(triplet_pairs)
        if args.show_triplet:
            show_images =  (images_all*128.+127.5)/255.
            save_dir = 'rm/{}_{}'.format(epoch,batch_number)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i in range(triplet_images_size//3):
                start_i = i*3
                image_a = show_images[triplet_pairs[start_i]]
                image_p = show_images[triplet_pairs[start_i+1]]
                image_n = show_images[triplet_pairs[start_i+2]]
                image_apn = np.concatenate([image_a,image_p,image_n],axis=1)
                to_name = '{}/{}.jpg'.format(save_dir,i)
                misc.imsave(to_name,image_apn)
        total_batch_size = args.num_gpus*args.people_per_batch*args.images_per_person
        nrof_batches = int(math.ceil(1.0 * (triplet_images_size//(args.num_gpus*3))*args.num_gpus*3 / total_batch_size))
        if nrof_batches == 0:
            print('continue forward')
            continue
        for i in range(nrof_batches):
            start_index = i*total_batch_size
            end_index = min((i+1)*total_batch_size, (triplet_images_size//(args.num_gpus*3))*args.num_gpus*3)
            #select_triplet_pairs = triplet_pairs[:total_batch_size] if triplet_images_size >= total_batch_size else triplet_pairs[:(triplet_images_size//(args.num_gpus*3))*args.num_gpus*3]
            select_triplet_pairs = triplet_pairs[start_index:end_index]
            select_images = images_all[select_triplet_pairs]
            #print('triplet pairs: {}/{}'.format(len(select_triplet_pairs)//3,triplet_images_size//3))
            print('triplet pairs: {}/{}'.format(end_index//3,triplet_images_size//3))

            start_time = time.time()
            print('Running forward pass on sampled images: ', end='')
            feed_dict = { phase_train_placeholder: False,images: select_images,learning_rate_placeholder:lr}
            start_time = time.time()
            triplet_err,total_err, _, step,lr,_, pos_np, neg_np = sess.run([loss['triplet_loss'],loss['total_loss'], train_op, global_step, learning_rate, summary_op, pos_d, neg_d], feed_dict=feed_dict)
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tTriplet Loss %2.3f Total Loss %2.3f lr %2.5f, pos_d %2.5f, neg_d %2.5f' %
                      (epoch, batch_number+1, args.epoch_size, duration, triplet_err,total_err,lr, pos_np,neg_np))
        # Add validation loss and accuracy to summary
        summary = tf.Summary()
        #pylint: disable=maybe-no-member
        summary.value.add(tag='time/selection', simple_value=duration)
        summary.value.add(tag='loss/triploss',simple_value=triplet_err)
        summary.value.add(tag='loss/total',simple_value=total_err)
        summary.value.add(tag='learning_rate/lr',simple_value=lr)
        summary_writer.add_summary(summary, step)
        
        batch_number += 1
        #with open('prefetch_cpu_var_2_{}.json'.format(batch_number),'w') as f:
        #    f.write(ctf)
    return step

def train_online(args, sess, epoch,
          learning_rate_placeholder, phase_train_placeholder, global_step, 
          loss, train_op, summary_op, summary_writer, learning_rate_schedule_file):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = utils.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        start_time = time.time()
        
        print('Running forward pass on sampled images: ', end='')
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True}
        start_time = time.time()
        triplet_err,total_err, _, step = sess.run([loss['triplet_loss'],loss['total_loss'], train_op, global_step ], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tTriplet Loss %2.3f Total Loss %2.3f lr %2.5f' %
                  (epoch, batch_number+1, args.epoch_size, duration, triplet_err,total_err,lr))
        #ctf = tl.generate_chrome_trace_format()
        batch_number += 1
    return step

def select_triplets_hardest(distances, labels, alpha):
    """ Select the triplets for training
    """
    #nrof_image_per_class = Counter(labels)
    time_start = time.time()
    label_counts = Counter(labels)
    nrof_images_per_class = [label_counts[l_ind] for l_ind in sorted(label_counts.keys())]
    #embeddings = embeddings.squeeze()
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    MAX=100000.


    for i in range(len(nrof_images_per_class)):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            #neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            neg_dists_sqr = distances[a_idx,:].copy() # a bug occur if we don't use copy. because the code bellow will assign them to np.NaN 

            p_d = -1
            p_i = -1
            for pair in range(j, nrof_images):
                p_idx = emb_start_idx + pair
                pos_dist_sqr = distances[a_idx, p_idx]
                if pos_dist_sqr > p_d:
                    p_i = p_idx
                    p_d = pos_dist_sqr
            
            neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = MAX
            n_idx = neg_dists_sqr.argmin()
            triplets.append([a_idx,p_i,n_idx])

        emb_start_idx += nrof_images
    trip_idx = len(triplets)
    time_select = time.time() - time_start
    print('time select triplet is {}'.format(time_select))
    print('nrof_choice_pairs {}'.format(trip_idx))
    triplet_inds = list(range(trip_idx))
    np.random.shuffle(triplet_inds)
    triplets = np.array(triplets,dtype=np.int64)
    triplets = np.hstack(triplets[triplet_inds])
    return triplets

def select_triplets_min_min(distances, labels, alpha):
    """ Select the triplets for training
    """
    #nrof_image_per_class = Counter(labels)
    time_start = time.time()
    label_counts = Counter(labels)
    #nrof_images_per_class = [label_counts[l_ind] for l_ind in sorted(label_counts.keys())]
    label_seen = set()
    nrof_images_per_class = []
    for l in labels:
        if l in label_seen:
            continue
        label_seen.add(l)
        nrof_images_per_class.append(label_counts[l])

    #embeddings = embeddings.squeeze()
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []


    for i in range(len(nrof_images_per_class)):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            #neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            neg_dists_sqr = distances[a_idx,:].copy() # a bug occur if we don't use copy. because the code bellow will assign them to np.NaN 
            neg_dist_tmp = 100 # max_dist
            triplet_tmp = []
            for pair in range(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                #pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                pos_dist_sqr = distances[a_idx, p_idx]
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    sort_inds = neg_dists_sqr[all_neg].argsort()
                    n_idx = all_neg[sort_inds[0]]
                    if neg_dists_sqr[n_idx] < neg_dist_tmp:
                        neg_dist_tmp = neg_dists_sqr[n_idx]
                        triplet_tmp = [a_idx, p_idx, n_idx]
                    num_trips += 1
            if len(triplet_tmp) > 0:
                triplets.append(triplet_tmp)
                trip_idx += 1

        emb_start_idx += nrof_images
    time_select = time.time() - time_start
    print('time select triplet is {}'.format(time_select))
    print('nrof_random_pairs {} nrof_choice_pairs {}'.format(num_trips,trip_idx))
    triplet_inds = list(range(trip_idx))
    np.random.shuffle(triplet_inds)
    triplets = np.array(triplets,dtype=np.int64)
    triplets = np.hstack(triplets[triplet_inds])
    return triplets


def select_triplets_min_max(distances, labels, alpha):
    """ Select the triplets for training
    """
    #nrof_image_per_class = Counter(labels)
    time_start = time.time()
    label_counts = Counter(labels)
    #nrof_images_per_class = [label_counts[l_ind] for l_ind in sorted(label_counts.keys())]
    #embeddings = embeddings.squeeze()
    label_seen = set()
    nrof_images_per_class = []
    for l in labels:
        if l in label_seen:
            continue
        label_seen.add(l)
        nrof_images_per_class.append(label_counts[l])


    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []


    for i in range(len(nrof_images_per_class)):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            #neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            neg_dists_sqr = distances[a_idx,:].copy() # a bug occur if we don't use copy. because the code bellow will assign them to np.NaN 
            neg_dist_tmp = -100 # max_dist
            triplet_tmp = []
            for pair in range(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                #pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                pos_dist_sqr = distances[a_idx, p_idx]
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    sort_inds = neg_dists_sqr[all_neg].argsort()
                    n_idx = all_neg[sort_inds[0]]
                    if neg_dists_sqr[n_idx] > neg_dist_tmp:
                        neg_dist_tmp = neg_dists_sqr[n_idx]
                        triplet_tmp = [a_idx, p_idx, n_idx]
                    num_trips += 1
            if len(triplet_tmp) > 0:
                triplets.append(triplet_tmp)
                trip_idx += 1

        emb_start_idx += nrof_images
    time_select = time.time() - time_start
    print('time select triplet is {}'.format(time_select))
    print('nrof_random_pairs {} nrof_choice_pairs {}'.format(num_trips,trip_idx))
    triplet_inds = list(range(trip_idx))
    np.random.shuffle(triplet_inds)
    triplets = np.array(triplets,dtype=np.int64)
    triplets = np.hstack(triplets[triplet_inds])
    return triplets



def select_triplets_batch_random(distances, labels, alpha):
    """ Select the triplets for training
    """
    #nrof_image_per_class = Counter(labels)
    time_start = time.time()
    label_counts = Counter(labels)
    nrof_images_per_class = [label_counts[l_ind] for l_ind in sorted(label_counts.keys())]
    #embeddings = embeddings.squeeze()
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []
    for i in range(len(nrof_images_per_class)):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            #neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            #pdb.set_trace()
            neg_dists_sqr = distances[a_idx,:].copy() # a bug occur if we don't use copy. because the code bellow will assign them to np.NaN 
            neg_dist_tmp = 100 # max_dist
            triplet_tmp = []
            for pair in range(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                #pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                pos_dist_sqr = distances[a_idx, p_idx]
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    #sort_inds = neg_dists_sqr[all_neg].argsort()
                    #n_idx = all_neg[sort_inds[0]]
                    triplet_tmp = [a_idx, p_idx, n_idx]
                    triplets.append(triplet_tmp)
                    num_trips += 1

        emb_start_idx += nrof_images
    time_select = time.time() - time_start
    print('time select triplet is {}'.format(time_select))
    print('nrof_random_pairs {}'.format(num_trips))
    triplet_inds = list(range(len(triplets)))
    np.random.shuffle(triplet_inds)
    triplets = np.array(triplets,dtype=np.int64)
    triplets = np.hstack(triplets[triplet_inds])
    return triplets

def select_triplets_batch_all(distances, labels, alpha):
    """ Select the triplets for training
    """
    #nrof_image_per_class = Counter(labels)
    time_start = time.time()
    label_counts = Counter(labels)
    nrof_images_per_class = [label_counts[l_ind] for l_ind in sorted(label_counts.keys())]
    #embeddings = embeddings.squeeze()
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []


    for i in range(len(nrof_images_per_class)):
        nrof_images = int(nrof_images_per_class[i])
        for j in range(1,nrof_images):
            a_idx = emb_start_idx + j - 1
            #neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            #pdb.set_trace()
            neg_dists_sqr = distances[a_idx,:].copy() # a bug occur if we don't use copy. because the code bellow will assign them to np.NaN 
            neg_dist_tmp = 100 # max_dist
            triplet_tmp = []
            for pair in range(j, nrof_images): # For every possible positive pair.
                p_idx = emb_start_idx + pair
                #pos_dist_sqr = np.sum(np.square(embeddings[a_idx]-embeddings[p_idx]))
                pos_dist_sqr = distances[a_idx, p_idx]
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                #all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where(neg_dists_sqr-pos_dist_sqr<alpha)[0] # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs>0:
                    triplet_tmp = [[a_idx, p_idx, n_idx] for n_idx in all_neg]
                    triplets.extend(triplet_tmp)
                    num_trips += len(triplet_tmp)

        emb_start_idx += nrof_images
    time_select = time.time() - time_start
    print('time select triplet is {}'.format(time_select))
    total_triplets = len(triplets)
    clip_trips = min(trip_thresh,total_triplets)
    print('nrof_random_pairs {} and clip trips {}'.format(num_trips,clip_trips))
    triplet_inds = list(range(len(triplets)))
    np.random.shuffle(triplet_inds)
    triplets = np.array(triplets,dtype=np.int64)
    triplets = np.hstack(triplets[triplet_inds])
    triplets = triplets[:clip_trips]
    return triplets
 
sample_index = 0
def sample_people(dataset, class_indices, people_per_batch, images_per_person):
    global sample_index
    nrof_images = people_per_batch * images_per_person

    nrof_classes = len(class_indices)
  
    # Sample classes from the dataset
    
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    #pdb.set_trace()
    # Sample images from these classes until we have enough
    if sample_index == 0:
        np.random.shuffle(class_indices)
    while len(image_paths)<nrof_images:
        if sample_index >= nrof_classes:
            sample_index = 0
            np.random.shuffle(class_indices)
        class_index = class_indices[sample_index]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images-len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        #print(idx,image_indices)
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        #print('class_index',class_index)
        sampled_class_indices += [class_index]*nrof_images_from_class
        image_paths += image_paths_for_class
        #num_per_class.append(nrof_images_from_class)
        sample_index+=1
  
    #return image_paths, num_per_class
    return image_paths, sampled_class_indices



def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  
  
def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='logs/facenet_ms_mp')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='models/facenet_ms_mp')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=.9)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.',
                        default='/home/ma-user/modelarts/user-job-dir/MassFac/train/models/facenet_ms_mp/20210923-160043/model-20210923-160043.ckpt-60')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
                        default='/home/ma-user/modelarts/user-job-dir/MassFac/dataset/casia-112x112')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.inception_resnet_v1')
    #!20000
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=10)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--image_src_size', type=int,
                        help='Src Image size (height, width) in pixels.', default=256)
    parser.add_argument('--people_per_batch', type=int,
                        help='Number of people per batch.', default=41)
    parser.add_argument('--num_gpus', type=int,
                        help='Number of gpus.', default=1)
    parser.add_argument('--scale', type=int,
                        help='scale batch of forward .', default=10)
    parser.add_argument('--images_per_person', type=int,
                        help='Number of images per person.', default=5)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=60)
    parser.add_argument('--alpha', type=float,
                        help='Positive to negative triplet distance margin.', default=0.25)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=1024)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--show_triplet',
                        help='show the select triplet pair', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=1e-4)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM', 'SGD'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.001)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=10)#!100
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=6686)
    # help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='data/learning_rate_schedule.txt')

    parser.add_argument('--network', type=str,
                        help='Which network to use.', default='mobilenet')
    parser.add_argument('--strategy', type=str,
                        help='triplet strategy to use.', default='min_and_max')
    parser.add_argument('--mine_method', type=str,
                        help='hard example mine method to use.', default='online')
    parser.add_argument('--dataset', type=str,
                        help='Which dataset used to train model', default='webface')
    parser.add_argument('--fc_bn',
                        help='Wheater use bn after fc.', action='store_true')
    parser.add_argument('--with_softmax',
                        help='Wheater train triplet with softmax loss.', action='store_true')
    parser.add_argument('--pretrain_softmax',
                        help='Wheater pretrain softmax loss first.', action='store_true')
    parser.add_argument('--softmax_loss_weight', type=float,
                        help='Softmax loss weight in total loss.', default=1.0)
    parser.add_argument('--softmax_epoch', type=int,
                        help='Softmax epoch to pretrain softmax.', default=5)
    parser.add_argument("--train_url",type=str,default="./output")
    parser.add_argument("--data_url",type=str,default="./dataset")
    parser.add_argument("--modelarts_data_dir", type=str, default="/home/ma-user/modelarts/user-job-dir/MassFac/dataset/")
    parser.add_argument("--modelarts_result_dir", type=str, default="/home/ma-user/modelarts/user-job-dir/MassFac/result")

    return parser.parse_args(argv)


if __name__ == '__main__':
    #mox.file.copy_parallel(src_url="obs://qyy-unet/MassFace-master/dataset/casia-112x112/", dst_url="data")
    code_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()
    print("===>>>code_dir:{},work_dir:{}".format(code_dir,work_dir))
    print(os.listdir(os.getcwd()))

    config = parse_arguments(sys.argv[1:])
    print("--------config--------")
    for k in list(vars(config).keys()):
        print("keys:{}:value:{}".format(k, vars(config)[k]))
    print("--------config--------")

    #if not os.path.exists(config.modelarts_result_dir):
        #os.makedirs(config.modelarts_result_dir)
        #bash_header = os.path.join(code_dir, 'train_triplet.sh')
        #arg__url = '%s %s %s %s' % (code_dir, config.modelarts_data_dir, config.modelarts_result_dir, config.train_url)
        #bash_command = 'bash %s %s' % (bash_header, arg__url)
        #print("bash command:", bash_command)
        #os.system(bash_command)

    main(parse_arguments(sys.argv[1:]))
    print("start to copy models!")
    mox.file.copy_parallel(src_url="models/facenet_ms_mp/", dst_url="obs://qyy-unet/MassFace-master/train/models/facenet_ms_mp/")
    #mox.file.copy_parallel(src_url="models/facenet_ms_mp/", dst_url="obs://2021-buckets-test2/MassFac/facenet_ms_mp/")

    print("copy down!")