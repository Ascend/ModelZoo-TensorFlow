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

import zipfile
from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
import importlib
import argparse
import facenet
import lfw
import h5py
import math
from circle_loss import *
import moxing as mox

def fully_connected(x, units, activation_fn=None,reuse=False):
    out=tf.layers.dense(inputs=x,units=units,activation=activation_fn,reuse=reuse)
    return out

def getStepTime():
    fin=open("step_time_list.txt","r")
    tot=0.0
    cnt=0.0
    for line in fin.readlines():
        line=float(line)
        tot+=line
        cnt+=1
    return tot/cnt
    fin.close()

def load_and_preprocess_image(img_path):
    img_raw = tf.io.read_file(img_path)
    img_tensor = tf.cond(
        tf.image.is_jpeg(img_raw),
        lambda: tf.image.decode_jpeg(img_raw,3),
        lambda: tf.image.decode_png(img_raw,3))
    img_tensor = tf.image.resize(img_tensor, [112, 112])
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = (img_tensor - 127.5) / 128.0
    return img_tensor

def getIteration(args, image_paths_placeholder, labels_placeholder):
    image_paths_dataset=tf.data.Dataset.from_tensor_slices(image_paths_placeholder).map(load_and_preprocess_image)
    label_dataset=tf.data.Dataset.from_tensor_slices(labels_placeholder)
    dataset=tf.data.Dataset.zip((image_paths_dataset,label_dataset)).prefetch(
        args.batch_size).batch(args.batch_size, drop_remainder=True)
    iteration=dataset.make_initializable_iterator()
    return iteration

def main(args):
    network = importlib.import_module(args.model_def)
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    stat_file_name = os.path.join(log_dir, 'stat.h5')

    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    dataset = facenet.get_dataset(args.data_dir)
    if args.filter_filename:
        dataset = filter_dataset(dataset, os.path.expanduser(args.filter_filename),
                                 args.filter_percentile, args.filter_min_nrof_images_per_class)

    if args.validation_set_split_ratio > 0.0:
        train_set, val_set = facenet.split_dataset(dataset, args.validation_set_split_ratio,
                                                   args.min_nrof_val_images_per_class, 'SPLIT_IMAGES')
    else:
        train_set, val_set = dataset, []

    nrof_classes = len(train_set)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)

    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        assert len(image_list) > 0, 'The training set should not be empty'

        val_image_list, val_label_list = facenet.get_image_paths_and_labels(val_set)

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder=tf.placeholder(tf.string,shape=(None,),name="image_paths")
        labels_placeholder=tf.placeholder(tf.int32,shape=(None,),name="labels")

        iteration = getIteration(args, image_paths_placeholder, labels_placeholder)
        image_batch,label_batch=iteration.get_next()

        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')

        print('Number of classes in training set: %d' % nrof_classes)
        print('Number of examples in training set: %d' % len(image_list))

        print('Number of classes in validation set: %d' % len(val_set))
        print('Number of examples in validation set: %d' % len(val_image_list))

        print('Building training graph')

        prelogits, _ = network.inference(image_batch, args.keep_probability,
                                         phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                                         weight_decay=args.weight_decay)
        logits=fully_connected(prelogits,len(train_set),None,False)

        norm_feat = tf.nn.l2_normalize(logits, 1, 1e-10, name='norm_feat')

        circle_loss_mean = circle_loss(tf.one_hot(label_batch, len(train_set), dtype=tf.float32),norm_feat,batch_size=args.batch_size)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')


        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=args.prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.prelogits_norm_loss_factor)

        prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size,
                                                   args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        tf.add_to_collection('losses', circle_loss_mean)

        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        total_loss = tf.add_n([circle_loss_mean] + regularization_losses, name='total_loss')

        train_op = facenet.train(total_loss, global_step, args.optimizer,
                                 learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)

        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)

        summary_op = tf.summary.merge_all()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)

        config_proto = tf.ConfigProto(gpu_options=gpu_options)
        custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        config = npu_config_proto(config_proto=config_proto)

        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # interaction_table.init.run()

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)

            print('Running training')
            nrof_steps = args.max_nrof_epochs * args.epoch_size
            nrof_val_samples = int(math.ceil(
                args.max_nrof_epochs / args.validate_every_n_epochs))
            stat = {
                'loss': np.zeros((nrof_steps,), np.float32),
                'center_loss': np.zeros((nrof_steps,), np.float32),
                'reg_loss': np.zeros((nrof_steps,), np.float32),
                'xent_loss': np.zeros((nrof_steps,), np.float32),
                'prelogits_norm': np.zeros((nrof_steps,), np.float32),
                'accuracy': np.zeros((nrof_steps,), np.float32),
                'val_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_xent_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_accuracy': np.zeros((nrof_val_samples,), np.float32),
                'lfw_accuracy': np.zeros((args.max_nrof_epochs,), np.float32),
                'lfw_valrate': np.zeros((args.max_nrof_epochs,), np.float32),
                'learning_rate': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_train': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_validate': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_evaluate': np.zeros((args.max_nrof_epochs,), np.float32),
                'prelogits_hist': np.zeros((args.max_nrof_epochs, 1000), np.float32),
            }

            best_acc=0.0
            for epoch in range(1, args.max_nrof_epochs + 1):
                step = sess.run(global_step, feed_dict=None)
                t = time.time()

                cont = train_epoch(args, sess,epoch, iteration,image_list,label_list,image_paths_placeholder,labels_placeholder,
                                learning_rate_placeholder, phase_train_placeholder, global_step,
                                total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file,
                                stat, circle_loss_mean, accuracy,
                                learning_rate, prelogits, prelogits_center_loss, args.random_rotate, args.random_crop,
                                args.random_flip, prelogits_norm,
                                args.prelogits_hist_max, args.use_fixed_image_standardization)

                stat['time_train'][epoch - 1] = time.time() - t
                if not cont:
                    break

                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, epoch)

                t = time.time()
                if args.lfw_dir:
                    acc_mean,acc_var=evaluate(sess,iteration,lfw_paths, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
                                 embeddings,label_batch,actual_issame, args.lfw_batch_size,
                                 args.lfw_nrof_folds, log_dir, step, summary_writer,stat,epoch,
                                 args.lfw_distance_metric, args.lfw_subtract_mean, args.lfw_use_flipped_images,
                                 args.use_fixed_image_standardization)
                    if acc_mean>best_acc:
                        best_acc=acc_mean
                stat['time_evaluate'][epoch - 1] = time.time() - t

                print('Saving statistics')
                with h5py.File(stat_file_name, 'w') as f:
                    for key, value in stat.items():
                        f.create_dataset(key, data=value)
                saveResultToOBS(args.train_url)

            fout=open("StepTime.txt","w")
            fout.write(str(getStepTime())+"\n")
            fout.close()

            fout=open("train_accuracy.txt","w")
            fout.write(str(best_acc)+"\n")
            fout.close()


    return model_dir

def train_epoch(args,sess,epoch,iteration,image_list,label_list,image_paths_placeholder,labels_placeholder,
                learning_rate_placeholder,phase_train_placeholder,step,
                loss,train_op,summary_op,summary_writer,reg_losses,learning_rate_schedule_file,
                stat,circle_loss_mean,accuracy,
                learning_rate,prelogits,prelogits_center_loss,random_rotate,random_crop,random_flip,prelogits_norm,
                prelogits_hist_max,use_fixed_image_standardization):

    epoch_idx=np.array(random.sample(range(len(label_list)),args.epoch_size*args.batch_size))
    image_list=np.array(image_list)[epoch_idx]
    label_list=np.array(label_list)[epoch_idx]

    batch_number = 0
    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    if lr <= 0:
        return False
    print("learning rate:",str(lr))
    feed_dict={image_paths_placeholder:image_list,
               labels_placeholder:label_list}
    sess.run(iteration.initializer,feed_dict=feed_dict)

    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()

        feed_dict = {learning_rate_placeholder: lr,
                     phase_train_placeholder: True}

        tensor_list = [loss, train_op, step, reg_losses, prelogits, circle_loss_mean, learning_rate, prelogits_norm,
                       accuracy, prelogits_center_loss]
        if batch_number % 100 == 0:
            loss_, _, step_, reg_losses_, prelogits_, circle_loss_mean_, lr_, prelogits_norm_, accuracy_, center_loss_, summary_str = sess.run(tensor_list + [summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step_)
        else:
            loss_, _, step_, reg_losses_, prelogits_, circle_loss_mean_, lr_, prelogits_norm_, accuracy_, center_loss_ = sess.run(tensor_list, feed_dict=feed_dict)

        stat['loss'][step_ - 1] = loss_
        stat['center_loss'][step_ - 1] = center_loss_
        stat['reg_loss'][step_ - 1] = np.sum(reg_losses_)
        stat['xent_loss'][step_ - 1] = circle_loss_mean_
        stat['prelogits_norm'][step_ - 1] = prelogits_norm_
        stat['learning_rate'][epoch - 1] = lr_
        stat['accuracy'][step_ - 1] = accuracy_
        stat['prelogits_hist'][epoch - 1, :] += \
            np.histogram(np.minimum(np.abs(prelogits_), prelogits_hist_max), bins=1000, range=(0.0, prelogits_hist_max))[0]

        duration = time.time() - start_time
        # print(
        #     'Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f\tCl %2.3f' %
        #     (epoch, batch_number + 1, args.epoch_size, duration, loss_, circle_loss_mean_, np.sum(reg_losses_),
        #      accuracy_, lr_, center_loss_))
        print("epoch : {}----step : {}----loss : {}----sec/step : {}".format(epoch, step_, loss_, duration))

        fout=open("step_time_list.txt","a+")
        fout.write(str(duration)+"\n")
        fout.close()

        batch_number += 1
        train_time += duration
    summary = tf.Summary()
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)
    return True

def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    threshold = np.interp(percentile * 0.01, cdf, bin_centers)
    return threshold


def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename, 'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center >= distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths) < min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del (filtered_dataset[i])

    return filtered_dataset

def evaluate(sess,iteration,lfw_paths,image_paths_placeholder,labels_placeholder,
             phase_train_placeholder,
             embeddings,label_batch, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer,stat, epoch,
             distance_metric, subtract_mean, use_flipped_images,
             use_fixed_image_standardization):
    start_time = time.time()
    print('Runnning forward pass on LFW images')

    nrof_embeddings = len(actual_issame) * 2
    nrof_images = nrof_embeddings

    embedding_size = int(embeddings.get_shape()[1])

    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    feed_dict={image_paths_placeholder:lfw_paths,
               labels_placeholder:np.array(range(len(lfw_paths)))}
    sess.run(iteration.initializer,feed_dict=feed_dict)

    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder: False}
        emb,lab= sess.run([embeddings,label_batch], feed_dict=feed_dict)

        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_embeddings, embedding_size))
    if use_flipped_images:
        embeddings[:, :embedding_size] = emb_array[0::2, :]
        embeddings[:, embedding_size:] = emb_array[1::2, :]
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(
        nrof_images)) == True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    _, _, accuracy, val, val_std, far = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds,
                                                     distance_metric=distance_metric, subtract_mean=subtract_mean)
    fout = open("result.txt", "a+")
    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    summary = tf.Summary()
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, 'lfw_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))
    stat['lfw_accuracy'][epoch - 1] = np.mean(accuracy)
    stat['lfw_valrate'][epoch - 1] = val
    return (np.mean(accuracy), np.std(accuracy))

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
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
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus',type=int,default=1,
                        help="the number of gpu")
    parser.add_argument('--data_url', type=str, default='obs://cann-id1221/dataset/',
                        help='the training data')
    parser.add_argument('--train_url', type=str, default='obs://cann-id1221/npu/',
                        help='the path model saved')
    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='./output/logs')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='./output/models')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.9)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.',default="")
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='./dataset/MS-Celeb-1M_clean_align')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.resnet34')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=1)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=30)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=2000)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=512)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate',
                        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization',
                        help='Performs fixed standardization of images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
                        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
                        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
                        help='Loss based on the norm of the activations in the prelogits layer.', default=0.0)
    parser.add_argument('--prelogits_norm_p', type=float,
                        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--prelogits_hist_max', type=float,
                        help='The max value for the prelogits histogram.', default=10.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM',"GD"],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=300)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=0.1)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='./data/learning_rate_circle_loss.txt')
    parser.add_argument('--filter_filename', type=str,
                        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
                        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
                        help='Keep only the classes with this number of examples or more', default=0)
    parser.add_argument('--validate_every_n_epochs', type=int,
                        help='Number of epoch between validation', default=5)
    parser.add_argument('--validation_set_split_ratio', type=float,
                        help='The ratio of the total dataset to use for validation', default=0.0)
    parser.add_argument('--min_nrof_val_images_per_class', type=float,
                        help='Classes with fewer images will be removed from the validation set', default=0)

    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='./data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.', default='./dataset/lfw-deepfunneled_align')
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=30)
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--lfw_distance_metric', type=int,
                        help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--lfw_use_flipped_images',
                        help='Concatenates embeddings for the image and its horizontally flipped counterpart.',
                        action='store_true')
    parser.add_argument('--lfw_subtract_mean',
                        help='Subtract feature mean before calculating distance.', action='store_true')
    return parser.parse_args(argv)

def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')

def saveResultToOBS(train_url):
    print("save result to obs........")
    print("train_url:",train_url)
    dst_url=os.path.join(train_url,str(datetime.now()))
    path="./output"
    mox.file.copy_parallel(path, dst_url)
    print("finish saving.....")

def getDatasetFromOBS(args):
    src_url=os.path.join(args.data_url,"dataset.zip")
    dst_url=os.getcwd()+"/dataset.zip"
    if os.path.exists(dst_url):
        print("dataset.zip has existed")
    else:
        print("Getting dataset from obs.........")
        mox.file.copy(src_url=src_url, dst_url=dst_url)
    if os.path.exists(os.getcwd()+"/dataset"):
        print("dataset has existed")
    else:
        print("Unzip dataset.zip.......")
        unzip_file(dst_url,os.getcwd())
    print("Finish get dataset......")

    print("get data.zip....")
    src_url=os.path.join(args.data_url,"data.zip")
    dst_url=os.getcwd()+"/data.zip"
    if os.path.exists(dst_url):
        print("data.zip has existed")
    else:
        mox.file.copy(src_url=src_url, dst_url=dst_url)
    if os.path.exists(os.getcwd()+"/data"):
        print("data has existed")
    else:
        unzip_file(dst_url, os.getcwd())
    print("finish get data")

def getTrain_url(args):
    args.train_url=os.path.join(args.train_url,str(datetime.now()))
    if not mox.file.exists(args.train_url):
        mox.file.mk_dir(args.train_url)

if __name__ == '__main__':
    args=parse_arguments(sys.argv[1:])
    getDatasetFromOBS(args)
    main(args)
