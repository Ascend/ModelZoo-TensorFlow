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

'''
Tensorflow implementation for MobileFaceNet.
Author: aiboy.wei@outlook.com .
'''


from npu_bridge.npu_init import *
from utils.data_process import load_data, next_batch, parse_function
import nets.TinyMobileFaceNet as TinyMobileFaceNet
import nets.MobileFaceNet as MobileFaceNet
from losses.face_losses import cos_loss, triplet_loss, arcface_loss
from verification import evaluate
from scipy.optimize import brentq
from utils.common import train, _add_loss_summaries
from scipy import interpolate
from datetime import datetime
from sklearn import metrics
import tensorflow as tf
import numpy as np
import argparse
import time
import os
# import precision_tool.tf_config as npu_tf_config

slim = tf.contrib.slim


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--less_steps', default=0, type=int, help='train for a few steps')
    parser.add_argument('--arch_text', default="./arch/txt/MobileFaceNet_Arch.txt")
    parser.add_argument('--var_text', default="./arch/txt/trainable_var.txt")
    parser.add_argument('--max_epoch', default=10, help='epoch to train the network')
    parser.add_argument('--image_size', default=[112, 112], help='the image size')
    parser.add_argument('--num_output', default=85164, help='the train images number')
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--weight_decay', default=5e-5, help='L2 weight regularization.')
    parser.add_argument('--lr_schedule', help='Number of epochs for learning rate piecewise.', default=[1, 4, 6, 8])
    parser.add_argument('--train_batch_size', default=64, help='batch size to train network')
    parser.add_argument('--test_batch_size', type=int,
                        help='Number of images to process in a batch in the test set.', default=100)
    parser.add_argument('--eval_datasets', default=['lfw'], help='evluation datasets')
    parser.add_argument('--eval_db_path', default='./datasets/faces_ms1m_112x112', help='evluate datasets base path')
    parser.add_argument('--eval_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--tfrecords_file_path', default='./datasets/tfrecords')
    parser.add_argument('--summary_path', default='./output/summary', help='the summary file save path')
    parser.add_argument('--ckpt_path', default='./output/ckpt', help='the ckpt file save path')
    parser.add_argument('--ckpt_best_path', default='./output/ckpt_best', help='the best ckpt file save path')
    parser.add_argument('--log_file_path', default='./output/logs', help='the ckpt file save path')
    parser.add_argument('--saver_maxkeep', default=20, help='tf.train.Saver max keep ckpt files')
    parser.add_argument('--buffer_size', default=10000, help='tf dataset api buffer size')
    parser.add_argument('--summary_interval', default=400, help='interval to save summary')
    parser.add_argument('--ckpt_interval', default=2000, help='intervals to save ckpt file')
    parser.add_argument('--validate_interval', default=2000, type=int, help='intervals to save ckpt file')
    parser.add_argument('--show_info_interval', default=100, type=int, help='intervals to save ckpt file')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--model_type', default=0, help='MobileFaceNet or TinyMobileFaceNet')
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--log_device_mapping', default=False, help='show device placement log')
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.999)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
                        help='Loss based on the norm of the activations in the prelogits layer.', default=5e-5)
    parser.add_argument('--prelogits_norm_p', type=float,
                        help='Norm to use for prelogits norm loss.', default=1.0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    with tf.Graph().as_default():
        #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        #打印软件栈日志 
        # os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = "1"
        args = get_parser()

          
        # create log dir
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        log_dir = os.path.join(os.path.expanduser(args.log_file_path), subdir)
        if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
            os.makedirs(log_dir)

        # define global parameters
        
        global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        epoch = tf.Variable(name='epoch', initial_value=-1, trainable=False)
        # define placeholder
        # inputs = tf.placeholder(name='img_inputs', shape=[None, *args.image_size, 3], dtype=tf.float32)
        inputs = tf.placeholder(name='img_inputs',
                                shape=[None, args.image_size[0], args.image_size[1], 3],
                                dtype=tf.float32)
        labels = tf.placeholder(name='img_labels', shape=[None, ], dtype=tf.int64)
        phase_train_placeholder = tf.placeholder_with_default(tf.constant(False, dtype=tf.bool), \
            shape=None, name='phase_train')

        # parse tfrecord
        tfrecords_f = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
        dataset = tf.data.TFRecordDataset(tfrecords_f)
        dataset = dataset.map(parse_function)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(args.train_batch_size, drop_remainder=True)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

#         img_batch, label_batch = next_batch(batch_size=args.train_batch_size,
#                                             path=os.path.join(args.tfrecords_file_path, 'tran.tfrecords'))

        # prepare validate datasets
        ver_list = []
        ver_name_list = []
        for db in args.eval_datasets:
            print('begin db %s convert.' % db)
            data_set = load_data(db, args.image_size, args)
            ver_list.append(data_set)
            ver_name_list.append(db)

        # pretrained model path
        pretrained_model = None
        if args.pretrained_model:
            pretrained_model = os.path.expanduser(args.pretrained_model)
            print('Pre-trained model: %s' % pretrained_model)

        # identity the input, for inference
        inputs = tf.identity(inputs, 'input')

        w_init_method = slim.initializers.xavier_initializer()
        if args.model_type == 0:
            prelogits, net_points = MobileFaceNet.inference(images=inputs,
                                                            phase_train=phase_train_placeholder,
                                                            weight_decay=args.weight_decay)
        else:
            prelogits, net_points = TinyMobileFaceNet.inference(images=inputs,
                                                                phase_train=phase_train_placeholder,
                                                                weight_decay=args.weight_decay)
        # record the network architecture
        hd = open(args.arch_text, 'w')
        for key in net_points.keys():
            info = '{}:{}\n'.format(key, net_points[key].get_shape().as_list())
            hd.write(info)
        hd.close()

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Norm for the prelogits
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits) + eps, ord=args.prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.prelogits_norm_loss_factor)

        inference_loss, logit = cos_loss(prelogits, labels, args.num_output)
#         inference_loss, logit = arcface_loss(embeddings, labels, args.num_output)
#         inference_loss, logit = triplet_loss(prelogits, labels, args.num_output, margin=0.5)

        tf.add_to_collection('losses', inference_loss)

        # total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([inference_loss] + regularization_losses, name='total_loss')


        # define the learning rate schedule
        learning_rate = tf.train.piecewise_constant(epoch,
                                                    boundaries=args.lr_schedule,
                                                    values=[0.01, 0.001, 0.001, 0.0001, 0.00001],
                                                    name='lr_schedule')
        
        init=tf.global_variables_initializer()
        config = tf.ConfigProto()
        #config = npu_tf_config.session_dump_config(config, action='dump')
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        # 设置网络静态内存和最大动态内存
        custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(21*1024 * 1024 * 1024))
        # 设置变量内存
        custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(str(10*1024 * 1024 * 1024))
        custom_op.parameter_map["fusion_switch_file"].s = \
            tf.compat.as_bytes("/home/test_user07/MobileFaceNet_Tensorflow/fusion_switch.cfg")
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        sess = tf.Session(config=config)
        sess.run(init)
        
        # calculate accuracy
        pred = tf.nn.softmax(logit)
        correct_prediction = tf.cast(tf.equal(tf.argmax(pred, 1), tf.cast(labels, tf.int64)), tf.float32)
        Accuracy_Op = tf.reduce_mean(correct_prediction)

        # summary writer
        summary = tf.summary.FileWriter(args.summary_path, sess.graph)
        summaries = []
        # add train info to tensorboard summary
        summaries.append(tf.summary.scalar('inference_loss', inference_loss))
        summaries.append(tf.summary.scalar('total_loss', total_loss))
        summaries.append(tf.summary.scalar('leraning_rate', learning_rate))
        summary_op = tf.summary.merge(summaries)

        # train op
        train_op = train(total_loss, global_step, args.optimizer, learning_rate, args.moving_average_decay,
                         tf.global_variables(), summaries, args.log_histograms)
        
        
        inc_global_step_op = tf.assign_add(global_step, 1, name='increment_global_step')
        inc_epoch_op = tf.assign_add(epoch, 1, name='increment_epoch')

        # record trainable variable
        hd = open(args.var_text, "w")
        for var in tf.trainable_variables():
            hd.write(str(var))
            hd.write('\n')
        hd.close()

        # saver to load pretrained model or save model
        # MobileFaceNet_vars = [v for v in tf.trainable_variables() if v.name.startswith('MobileFaceNet')]
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=args.saver_maxkeep)

        #init all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load pretrained model
        if pretrained_model:
            print('Restoring pretrained model: %s' % pretrained_model)
            ckpt = tf.train.get_checkpoint_state(pretrained_model)
            print(ckpt)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # output file path
        if not os.path.exists(args.log_file_path):
            os.makedirs(args.log_file_path)
        if not os.path.exists(args.ckpt_best_path):
            os.makedirs(args.ckpt_best_path)

        total_accuracy = {}
        _ = sess.run(inc_epoch_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        best_acc = 0.986
        print('==============  start training ===============')
        for i in range(args.max_epoch):
            sess.run(iterator.initializer)
            count = 0
            while True:
                try:
                    images_train, labels_train = sess.run(next_element)
                    #images_train, labels_train = sess.run([img_batch, label_batch])

                    feed_dict = {inputs: images_train, labels: labels_train, phase_train_placeholder: True}
                    start = time.time()
                    
                    
                    _, total_loss_val, inference_loss_val, reg_loss_val, _, acc_val = \
                        sess.run([train_op, total_loss, inference_loss, regularization_losses, inc_global_step_op,
                                  Accuracy_Op],
                                 feed_dict=feed_dict)
            
                    end = time.time()
                    pre_sec = args.train_batch_size / (end - start)

                    count += 1
                    # print training information
                    if count > 0 and count % args.show_info_interval == 0:
                        print('epoch %d, step %d, total loss is: %.2f, inference loss is: %.2f, reg_loss is: %.2f' % \
                              (i, count, total_loss_val, inference_loss_val, np.sum(reg_loss_val)))
                        
                        with open(os.path.join(log_dir, 'trainlog.txt'), 'at') as f:
                                f.write('%d\t%.2f\t%.2f\t%.2f\n' % (count, total_loss_val, inference_loss_val, np.sum(reg_loss_val)))
                                
                    # save summary
                    if count > 0 and count % args.summary_interval == 0:
                        feed_dict = {inputs: images_train, labels: labels_train, phase_train_placeholder: True}
                        summary_op_val = sess.run(summary_op, feed_dict=feed_dict)
                        summary.add_summary(summary_op_val, count)

                    # save ckpt files
                    if count > 0 and count % args.ckpt_interval == 0:
                        filename = 'MobileFaceNet_epoch_{:d}_iter_{:d}'.format(i, count) + '.ckpt'
                        filename = os.path.join(args.ckpt_path, filename)
                        saver.save(sess, filename)

                    # validate
                    if count > 0 and count % args.validate_interval == 0:
                        print('\nIteration', count, 'testing...')
                        for ver_step in range(len(ver_list)):
                            start_time = time.time()
                            data_sets, issame_list = ver_list[ver_step]
                            emb_array = np.zeros((data_sets.shape[0], args.embedding_size))
                            nrof_batches = data_sets.shape[0] // args.test_batch_size
                            for index in range(nrof_batches):  # actual is same multiply 2, test data total
                                start_index = index * args.test_batch_size
                                end_index = min((index + 1) * args.test_batch_size, data_sets.shape[0])

                                feed_dict = {inputs: data_sets[start_index:end_index, ...], phase_train_placeholder: False}
                                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                            duration = time.time() - start_time
                            tpr, fpr, accuracy, val, val_std, far = evaluate(emb_array, issame_list, nrof_folds=args.eval_nrof_folds)

                            print('==============  Validation: accuracy on 12000 LFW images is: %.5f  ===============' % (np.mean(accuracy)))
                            
                            with open(os.path.join(log_dir, '{}_result.txt'.format(ver_name_list[ver_step])), 'at') as f:
                                f.write('%d\t%.5f\t%.5f\n' % (count, np.mean(accuracy), val))

                            if ver_name_list == 'lfw' and np.mean(accuracy) > best_acc:
                                best_acc = np.mean(accuracy)
                                filename = 'MobileFaceNet_best.ckpt'
                                filename = os.path.join(args.ckpt_best_path, filename)
                                saver.save(sess, filename)
                                tf.io.write_graph(sess.graph, args.ckpt_best_path, 'graph.pbtxt', as_text=True)
                    if count > 0 and args.train_batch_size * count > 3804846:
                        break
                        
                    # train for a few steps and stop
                    if args.less_steps != 0 and count > args.less_steps:
                        print('early stop!')
                        exit()
                        
                except tf.errors.OutOfRangeError:
                    print("End of epoch %d" % i)
                    break

        coord.request_stop()
        coord.join(threads)
        sess.close()
