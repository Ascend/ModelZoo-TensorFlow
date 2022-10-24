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
import tensorflow as tf
import numpy as np
import C3D_model
import data_processing
import os.path
import time
from npu_config import npu_config
from npu_bridge.npu_init import *

tf.logging.set_verbosity(tf.logging.ERROR)


flags = tf.flags
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
flags.DEFINE_string("result", "result", "train model result")
flags.DEFINE_string("train_model_url", "train_model_url", "save model checkpoint url")
flags.DEFINE_boolean("profiling", False, "model profiling start")
flags.DEFINE_string("data_dir", "data_dir", "dataset dir")
FLAGS = flags.FLAGS

TRAIN_LOG_DIR = FLAGS.result
TRAIN_CHECK_POINT = FLAGS.train_model_url
TRAIN_LIST_PATH = FLAGS.data_dir + '/train.list'
TEST_LIST_PATH = FLAGS.data_dir + '/test.list'
BATCH_SIZE = FLAGS.batch_size
NUM_CLASSES = 102
CROP_SZIE = 112
CHANNEL_NUM = 3
CLIP_LENGTH = 16
EPOCH_NUM = FLAGS.max_steps
LR_DECAY_FACTOR = 0.5
EPOCHS_PER_LR_DECAY = 2
MOVING_AV_DECAY = 0.9999

# Get shuffle index
train_video_indices, validation_video_indices = data_processing.get_video_indices(TRAIN_LIST_PATH)

RANK_SIZE = int(os.getenv('RANK_SIZE'))
rank_id = int(os.getenv('RANK_ID'))


with tf.Graph().as_default():
    batch_clips = tf.placeholder(tf.float32, [BATCH_SIZE, CLIP_LENGTH, CROP_SZIE, CROP_SZIE, CHANNEL_NUM], name='X')
    batch_labels = tf.placeholder(tf.int32, [BATCH_SIZE, NUM_CLASSES], name='Y')
    keep_prob = tf.placeholder(tf.float32)
    logits = C3D_model.C3D(batch_clips, NUM_CLASSES, keep_prob)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_labels))
        tf.summary.scalar('entropy_loss', loss)

    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(batch_labels, 1)), np.float32))
        tf.summary.scalar('accuracy', accuracy)
    # global_step = tf.Variable(0, name='global_step', trainable=False)
    # decay_step = EPOCHS_PER_LR_DECAY * len(train_video_indices) // BATCH_SIZE
    learning_rate = 1e-4
    if int(RANK_SIZE) > 1:
        optimizer = tf.train.AdamOptimizer(learning_rate) # .minimize(loss)  # , global_step=global_step)
        optimizer = npu_distributed_optimizer_wrapper(optimizer).minimize(loss)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)  # , global_step=global_step)
    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["hcom_parallel"].b = True
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭

    with tf.Session(config=config) as sess:
        train_summary_writer = tf.summary.FileWriter(TRAIN_LOG_DIR, sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if int(RANK_SIZE) > 1:
            input = tf.trainable_variables()
            bcast_global_variables_op = hccl_ops.broadcast(input, 0)
            sess.run(bcast_global_variables_op)
        step = 0
        for epoch in range(EPOCH_NUM):
            
            accuracy_epoch = 0
            loss_epoch = 0
            batch_index = 0+rank_id*BATCH_SIZE
            for i in range(len(train_video_indices) // (BATCH_SIZE*RANK_SIZE)):
                step += 1
                batch_data, batch_index = data_processing.get_batches(TRAIN_LIST_PATH, NUM_CLASSES, batch_index,
                                                                      train_video_indices, BATCH_SIZE, RANK_SIZE)
                train_time = time.time()
                _, loss_out, accuracy_out, summary = sess.run([optimizer, loss, accuracy, summary_op],
                                                            feed_dict={batch_clips: batch_data['clips'],
                                                                        batch_labels: batch_data['labels'],
                                                                        keep_prob: 0.5})
                loss_epoch += loss_out
                accuracy_epoch += accuracy_out
                print("%d Train Time = %.3f" % (step, time.time() - train_time))

                if i % 10 == 0:
                    # print('Epoch %d, Batch %d: Loss is %.5f; Accuracy is %.5f' % (epoch + 1, i, loss_out, accuracy_out))
                    train_summary_writer.add_summary(summary, step)
            print('Epoch %d: Average loss is: %.5f ; Average accuracy is: %.5f' % (
                epoch + 1, loss_epoch / (len(train_video_indices) // (BATCH_SIZE*RANK_SIZE)),
                accuracy_epoch / (len(train_video_indices) // (BATCH_SIZE*RANK_SIZE))))
            accuracy_epoch = 0
            loss_epoch = 0
            # batch_index = 0+rank_id*BATCH_SIZE
            batch_index=0
            for i in range(len(validation_video_indices) // (BATCH_SIZE)):
                batch_data, batch_index = data_processing.get_batches(TRAIN_LIST_PATH, NUM_CLASSES, batch_index,
                                                                      validation_video_indices, BATCH_SIZE, 1)
                loss_out, accuracy_out = sess.run([loss, accuracy],
                                                feed_dict={batch_clips: batch_data['clips'],
                                                            batch_labels: batch_data['labels'],
                                                            keep_prob: 1.0})
                loss_epoch += loss_out
                accuracy_epoch += accuracy_out

            print('Validation loss is %.5f ; Accuracy is %.5f' % (
                loss_epoch / (len(validation_video_indices) // (BATCH_SIZE)),
                accuracy_epoch / (len(validation_video_indices) // (BATCH_SIZE))))
            #print("%d Train Time = %.3f" % (epoch + 1, time.time() - train_time))
            saver.save(sess, TRAIN_CHECK_POINT + '/train.ckpt', global_step=epoch)

if(rank_id==RANK_SIZE-1):
    # test data
    test_num = data_processing.get_test_num(TEST_LIST_PATH)
    test_video_indices = range(test_num)

    with tf.Graph().as_default():
        batch_clips = tf.placeholder(tf.float32, [BATCH_SIZE, CLIP_LENGTH, CROP_SZIE, CROP_SZIE, CHANNEL_NUM], name='X')
        batch_labels = tf.placeholder(tf.int32, [BATCH_SIZE, NUM_CLASSES], name='Y')
        keep_prob = tf.placeholder(tf.float32)
        logits = C3D_model.C3D(batch_clips, NUM_CLASSES, keep_prob)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(batch_labels, 1)), np.float32))

        restorer = tf.train.Saver()
        config = npu_config(FLAGS)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # if int(RANK_SIZE) > 1:
            #     input = tf.trainable_variables()
            #     bcast_global_variables_op = hccl_ops.broadcast(input, 0)
            #     sess.run(bcast_global_variables_op)
            ckpt = tf.train.get_checkpoint_state(TRAIN_CHECK_POINT)
            if ckpt:
                restorer.restore(sess, ckpt.model_checkpoint_path)
            accuracy_epoch = 0
            batch_index = 0
            # batch_index = 0+rank_id*BATCH_SIZE
            for i in range(test_num // (BATCH_SIZE)):
                batch_data, batch_index = data_processing.get_batches(TEST_LIST_PATH, NUM_CLASSES, batch_index,
                                                                    test_video_indices, BATCH_SIZE, 1)
                accuracy_out = sess.run(accuracy, feed_dict={batch_clips: batch_data['clips'],
                                                            batch_labels: batch_data['labels'],
                                                            keep_prob: 1.0})
                accuracy_epoch += accuracy_out

        print('Test accuracy is %.5f' % (accuracy_epoch / (test_num // (BATCH_SIZE*1))))
