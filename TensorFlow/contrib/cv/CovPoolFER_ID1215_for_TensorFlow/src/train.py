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

from datetime import datetime
import os.path
import os
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import framework
import h5py
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from PIL import Image
from npu_bridge.npu_init import *

#鍒涘缓session
config = tf.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
#custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 蹇呴』鏄惧紡鍏抽棴
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 蹇呴』鏄惧紡鍏抽棴

# custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")
# dump_path锛歞ump鏁版嵁瀛樻斁璺緞锛岃鍙傛暟鎸囧畾鐨勭洰褰曢渶瑕佸湪鍚姩璁粌鐨勭幆澧冧笂锛堝鍣ㄦ垨Host渚э級鎻愬墠鍒涘缓涓旂‘淇濆畨瑁呮椂閰嶇疆#dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限
# custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/lf/src/train_url/")
# enable_dump_debug：是否开启溢出检测功能
# custom_op.parameter_map["enable_dump_debug"].b = True
# dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
# custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all")

# block the fusion rules
custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes("./precision_tool/fusion_switch.cfg")
# custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes("/home/lf/src/fusion_switch.cfg")
# mixed precision
# custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp32")

def main(args):

    network = importlib.import_module(args.model_def)
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    train_set = framework.get_dataset(args.data_dir)
    nrof_classes = len(train_set)
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)

    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Get a list of image paths and their labels
        image_list, label_list = framework.get_image_paths_and_labels(train_set)
        assert len(image_list) > 0, 'The dataset should not be empty'

        image_list = np.array([np.array(Image.open(path)).astype(np.float32) for path in image_list])

        label_list = np.array(label_list)

        traindb = tf.data.Dataset.from_tensor_slices((image_list, label_list)).shuffle(10).batch(args.batch_size,
                                                                                                 drop_remainder=True).repeat()
        traindb = traindb.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        assert len(image_list) > 0, 'The dataset should not be empty'
        # Create a queue that produces indices into the image_list and label_list
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)

        inputX = tf.placeholder(tf.float32, shape=[args.batch_size, args.image_size, args.image_size, 3],
                                name='inputImage')
        inputY = tf.placeholder(tf.int32, shape=[args.batch_size],
                                name='inputLabel')
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: %d' % len(image_list))
        print('Building training graph')
        prelogits = network.inference(inputX, args.keep_probability,
                                      bottleneck_layer_size=args.embedding_size,
                                      weight_decay=args.weight_decay)
        logits = slim.fully_connected(prelogits, nrof_classes, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(args.weight_decay),
                                      scope='Logits', reuse=False)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # Add center loss
        if args.center_loss_factor > 0.0:
            prelogits_center_loss, _ = framework.center_loss(prelogits, inputY, args.center_loss_alfa,
                                                             nrof_classes)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.learning_rate_decay_epochs * args.epoch_size,
                                                   args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=inputY, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op, redun = framework.trainspd(total_loss, global_step, args.optimizer,
                                             learning_rate, args.moving_average_decay, tf.global_variables(),
                                             args.log_histograms)

        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        iterator = traindb.make_initializable_iterator()
        data_iter = iterator.get_next()

        with sess.as_default():

            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)

            # Training and validation loop
            print('Running training')
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, epoch, iterator, data_iter , inputX, inputY,
                      learning_rate_placeholder, global_step,
                      total_loss, train_op, redun, summary_op, summary_writer, regularization_losses,
                      args.learning_rate_schedule_file)
                # Save variables and the metagraph if it doesn't exist already
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

    sess.close()
    return model_dir


def train(args, sess, epoch, iterator, data_iter, inputX, inputY, learning_rate_placeholder, global_step,
          loss, train_op, redun, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file):
    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = framework.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    train_time = 0
    sess.run(iterator.initializer)
    while batch_number < args.epoch_size:
        start_time = time.time()
        image_batch, label_batch = sess.run(data_iter)
        feed_dict = {learning_rate_placeholder: lr, inputX: image_batch, inputY: label_batch}
        if (batch_number % 100 == 0):
            err, _, _, step, reg_loss, summary_str = sess.run(
                [loss, train_op, redun, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, _, step, reg_loss = sess.run([loss, train_op, redun, global_step, regularization_losses],
                                                 feed_dict=feed_dict)
        duration = time.time() - start_time
        s = 'Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %(epoch, batch_number + 1, args.epoch_size, duration, err, np.sum(reg_loss))
        print(s)
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step

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
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_url", type=str)
    parser.add_argument("--data_url", type=str)
    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='~/logs/')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='~/models/')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
                        default='~/data/SFEW/Train')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.covpoolnet')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=128)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=100)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate',
                        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--decov_loss_factor', type=float,
                        help='DeCov loss factor.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
                        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
                        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Number of preprocessing (data loading and augumentation) threads.', default=4)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='data/learning_rate_schedule.txt')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
