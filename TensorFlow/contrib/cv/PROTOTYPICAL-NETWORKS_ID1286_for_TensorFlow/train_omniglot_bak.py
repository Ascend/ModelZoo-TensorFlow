#!/usr/bin/env python
# coding: utf-8
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
Training and Testing
"""

import numpy as np
import tensorflow as tf
from model import prototypical, euclidean_distance
from data import Data
from cfg import make_config
import os
os.environ['GE_USE_STATIC_MEMCRY'] = '1'
from tensorflow.python.client import timeline
import datetime


flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("result", "result", "The result directory where the model checkpoints will be written.")
flags.DEFINE_string("dataset", "dataset", "dataset path")
flags.DEFINE_string("obs_dir", "obs://prototypical/log", "obs result path, not need on gpu and apulis platform")

## Other parametersresult
flags.DEFINE_integer("n_epochs", 20, "number of epochs")
flags.DEFINE_integer("n_episodes", 100, "number of episodes")
flags.DEFINE_integer("n_way", 60, "number of ways")
flags.DEFINE_integer("n_shot", 5, "number of shots")
flags.DEFINE_integer("n_query", 5, "number of queries")
flags.DEFINE_integer("n_examples", 20, "number of examples")
flags.DEFINE_integer("im_width", 28, "width of image")
flags.DEFINE_integer("im_height", 28, "height of image")
flags.DEFINE_integer("channels", 1, "number of channels")

flags.DEFINE_integer("n_test_episodes", 1000, "number of test episodes")
flags.DEFINE_integer("n_test_way", 20, "number of test ways")
flags.DEFINE_integer("n_test_shot", 5, "number of test shots")
flags.DEFINE_integer("n_test_query", 15, "number of test queries")

flags.DEFINE_string("resume_path", None, "checkpoint path, /cache/checkpoint/model.ckpt for NPU,"
                                         "and ./checkpoint/model.ckpt for GPU")
flags.DEFINE_string("chip", "npu", "Run on which chip, (npu or gpu or cpu)")
flags.DEFINE_string("platform", "apulis", "Run on linux/apulis/modelarts platform."
                                          "Modelarts Platform has some extra data copy operations")
flags.DEFINE_boolean("profiling", False, "profiling for performance or not")

if FLAGS.chip == 'npu':
    from npu_bridge.npu_init import *

def main(_):
    """
    main
    """
    n_epochs = FLAGS.n_epochs
    n_episodes = FLAGS.n_episodes
    n_way = FLAGS.n_way
    n_shot = FLAGS.n_shot
    n_query = FLAGS.n_query
    n_examples = FLAGS.n_examples
    im_width = FLAGS.im_width
    im_height = FLAGS.im_height
    channels = FLAGS.channels
    h_dim = 64
    z_dim = 64

    n_test_episodes = FLAGS.n_test_episodes
    n_test_way = FLAGS.n_test_way
    n_test_shot = FLAGS.n_test_shot
    n_test_query = FLAGS.n_test_query
    # max_acc = 0.

    # training data
    train_data = Data(n_examples=n_examples, im_height=im_height, im_width=im_width, datatype='train.txt')
    print(train_data.shape)

    # testing data
    test_data = Data(n_examples=n_examples, im_height=im_height, im_width=im_width, datatype='test.txt')
    print(test_data.shape)

    x = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
    q = tf.placeholder(tf.float32, [None, None, im_height, im_width, channels])
    x_shape = tf.shape(x)
    q_shape = tf.shape(q)
    num_classes, num_support = x_shape[0], x_shape[1]
    num_queries = q_shape[1]
    y = tf.placeholder(tf.int64, [None, None])
    y_one_hot = tf.one_hot(y, depth=num_classes)

    emb_x = prototypical(x=tf.reshape(x, [num_classes * num_support, im_height, im_width, channels]), h_dim=h_dim,
                         z_dim=z_dim, reuse=False)
    emb_dim = tf.shape(emb_x)[-1]
    emb_x = tf.reduce_mean(tf.reshape(emb_x, [num_classes, num_support, emb_dim]), axis=1)

    emb_q = prototypical(tf.reshape(q, [num_classes * num_queries, im_height, im_width, channels]), h_dim=h_dim,
                         z_dim=z_dim, reuse=True)
    dists = euclidean_distance(emb_q, emb_x)
    log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes, num_queries, -1])
    ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(log_p_y, axis=-1), y), dtype=tf.float32))

    train_op = tf.train.AdamOptimizer().minimize(ce_loss)

    ## gpu profiling configuration
    if FLAGS.chip.lower() == 'gpu' and FLAGS.profiling:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    else:
        options = None
        run_metadata = None

    config = make_config(FLAGS)

    # start training
    sess = tf.InteractiveSession(config=config)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # saver is used to save the model
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    if FLAGS.resume_path is not None:
        # checkpoint_path = os.path.join('cache/checkpoint/', FLAGS.resume_path)
        tf.logging.info('Loading checkpoint from {}...'.format(FLAGS.resume_path))
        # model_file = tf.train.latest_checkpoint(FLAGS.resume_path)
        print("===>>>Model Files:", FLAGS.resume_path)
        saver.restore(sess, FLAGS.resume_path)
        # print("weights:", sess.run(q))

    # training
    print('Training...')
    for ep in range(n_epochs):
        # Computing the time for one epoch
        start = datetime.datetime.now()
        for epi in range(n_episodes):
            epi_classes = np.random.permutation(train_data.n_classes)[:n_way]
            support = np.zeros([n_way, n_shot, im_height, im_width], dtype=np.float32)
            query = np.zeros([n_way, n_query, im_height, im_width], dtype=np.float32)
            for i, epi_cls in enumerate(epi_classes):
                selected = np.random.permutation(n_examples)[:n_shot + n_query]
                support[i] = train_data.dataset[epi_cls, selected[:n_shot]]
                query[i] = train_data.dataset[epi_cls, selected[n_shot:]]
            support = np.expand_dims(support, axis=-1)
            query = np.expand_dims(query, axis=-1)
            labels = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
            _, ls, ac = sess.run([train_op, ce_loss, acc], feed_dict={x: support, q: query, y: labels})

            if (epi + 1) % 50 == 0:
                print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(ep + 1, n_epochs, epi + 1,
                                                                                         n_episodes, ls, ac))
        end = datetime.datetime.now()
        train_deltatime = (end - start).total_seconds() * 1000
        print('Time Used for One Epoch===>>>{:.3f} (ms)'.format(train_deltatime))
        # tf.logging.info("Time Used===>>>[FP+BP]:{:.3f} (ms)".format(train_deltatime))

    # test
    print('Testing...')
    avg_acc = 0.
    for epi in range(n_test_episodes):
        epi_classes_test = np.random.permutation(test_data.n_classes)[:n_test_way]
        support_test = np.zeros([n_test_way, n_test_shot, im_height, im_width], dtype=np.float32)
        query_test = np.zeros([n_test_way, n_test_query, im_height, im_width], dtype=np.float32)
        for i, epi_cls in enumerate(epi_classes_test):
            selected_test = np.random.permutation(n_examples)[:n_test_shot + n_test_query]
            support_test[i] = test_data.dataset[epi_cls, selected_test[:n_test_shot]]
            query_test[i] = test_data.dataset[epi_cls, selected_test[n_test_shot:]]
        support_test = np.expand_dims(support_test, axis=-1)
        query_test = np.expand_dims(query_test, axis=-1)
        labels_test = np.tile(np.arange(n_test_way)[:, np.newaxis], (1, n_test_query)).astype(np.uint8)
        ls_test, ac_test = sess.run([ce_loss, acc], feed_dict={x: support_test, q: query_test, y: labels_test})
        avg_acc += ac_test
        if (epi + 1) % 50 == 0:
            print('[test episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(epi + 1, n_test_episodes,
                                                                             ls_test, ac_test))
    avg_acc /= n_test_episodes
    print('Average Test Accuracy: {:.5f}'.format(avg_acc))

    # save model
    saver.save(sess=sess, save_path=os.path.join(FLAGS.result, "model.ckpt"))
    # saver.save(sess=sess, save_path="/cache/result/checkpoint/model.ckpt")
    print('Model has saved!')


    if FLAGS.chip.lower() == 'gpu' and FLAGS.profiling:
        work_dir = os.getcwd()
        timeline_path = os.path.join(work_dir, 'timeline.ctf.json')
        with open(timeline_path, 'w') as trace_file:
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            trace_file.write(trace.generate_chrome_trace_format())

    if FLAGS.platform.lower() == 'modelarts':
        from help_modelarts import modelarts_result2obs
        from help_modelarts import modelarts_checkpoint2obs
        modelarts_result2obs(FLAGS)
        modelarts_checkpoint2obs(FLAGS)

    sess.close()

if __name__ == "__main__":
    flags.mark_flag_as_required("dataset")
    flags.mark_flag_as_required("result")
    flags.mark_flag_as_required("obs_dir")
    tf.app.run()



