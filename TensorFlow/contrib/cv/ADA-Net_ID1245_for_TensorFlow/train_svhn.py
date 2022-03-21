# ==============================================================================
# MIT License

# Copyright (c) 2019 Qin Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

from npu_bridge.npu_init import *
import time

import numpy
import tensorflow as tf

import layers as L
import cnn

from flip_gradient import flip_gradient
from svhn import inputs, unlabeled_inputs

#import moxing as mox
import sys
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('device', '/gpu:0', "device")
tf.app.flags.DEFINE_string('dataset', 'cifar10', "{cifar10, svhn}")
tf.app.flags.DEFINE_string('logdir', "./svhn_log_1/", "logdir")
tf.app.flags.DEFINE_integer('seed', 1, "initial random seed")
tf.app.flags.DEFINE_bool('validation', False, "")
tf.app.flags.DEFINE_bool('one_hot', False, "")
tf.app.flags.DEFINE_integer('batch_size', 128, "the number of examples in a batch")
tf.app.flags.DEFINE_integer('ul_batch_size', 128, "the number of unlabeled examples in a batch")
tf.app.flags.DEFINE_integer('eval_batch_size', 100, "the number of eval examples in a batch")
tf.app.flags.DEFINE_integer('eval_freq', 5, "")
tf.app.flags.DEFINE_integer('num_epochs', 500, "the number of epochs for training")
tf.app.flags.DEFINE_integer('epoch_decay_start', 80, "epoch of starting learning rate decay")
tf.app.flags.DEFINE_integer('num_iter_per_epoch', 400, "the number of updates per epoch")
tf.app.flags.DEFINE_float('learning_rate', 0.0005, "initial leanring rate")
tf.app.flags.DEFINE_float('mom1', 0.9, "initial momentum rate")
tf.app.flags.DEFINE_float('mom2', 0.5, "momentum rate after epoch_decay_start")
tf.app.flags.DEFINE_string('data_url', './svhn/dataset', "data_root")
tf.app.flags.DEFINE_string('train_url', './output', "output_root")
tf.app.flags.DEFINE_integer('save_model_secs', 150, "the number of updates per epoch")

code_dir = os.path.dirname(__file__)
work_path = os.path.join(code_dir, '../')
sys.path.append(work_path)

NUM_EVAL_EXAMPLES = 5000

def logit(x, is_training=True, update_batch_stats=True, stochastic=True, seed=1234):
    return cnn.logit(x, is_training=is_training,
                     update_batch_stats=update_batch_stats,
                     stochastic=stochastic,
                     seed=seed)[0]


def forward(x, is_training=True, update_batch_stats=True, seed=1234):
    if is_training:
        return logit(x, is_training=True,
                     update_batch_stats=update_batch_stats,
                     stochastic=True, seed=seed)
    else:
        return logit(x, is_training=False,
                     update_batch_stats=update_batch_stats,
                     stochastic=False, seed=seed)


def build_training_graph(x1, y1, x2, lr, mom):
    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False,
    )
    k = 1. * global_step / (FLAGS.num_iter_per_epoch * FLAGS.num_epochs)
    # lp schedule from GRL
    lp = (2. / (1. + tf.exp(-10. * k)) - 1)

    # Interpolation 
    y2_logit, _ = cnn.logit(x2, is_training=False, update_batch_stats=False, stochastic=False)
    if FLAGS.one_hot:
        y2 = tf.stop_gradient(tf.cast(tf.one_hot(tf.argmax(y2_logit, -1), 10), tf.float32))
    else:
        y2 = tf.stop_gradient(tf.nn.softmax(y2_logit))

    dist_beta = tf.distributions.Beta(0.1, 0.1)
    lmb = dist_beta.sample(tf.shape(x1)[0])
    lmb_x = tf.reshape(lmb, [-1, 1, 1, 1])
    lmb_y = tf.reshape(lmb, [-1, 1])
    x = x1 * lmb_x + x2 * (1. - lmb_x)
    y = y1 * lmb_y + y2 * (1. - lmb_y)

    label_dm = tf.concat([tf.reshape(lmb, [-1, 1]), tf.reshape(1. - lmb, [-1, 1])], axis=1)

    # Calculate the feats and logits on interpolated samples
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        logit, net = cnn.logit(x, is_training=True, update_batch_stats=True)

    # Alignment Loss
    net_ = flip_gradient(net, lp)
    logitsdm = tf.layers.dense(net_, 1024, activation=tf.nn.relu, name='linear_dm1')
    logitsdm = tf.layers.dense(logitsdm, 1024, activation=tf.nn.relu, name='linear_dm2')
    logits_dm = tf.layers.dense(logitsdm, 2, name="logits_dm")
    dm_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_dm, logits=logits_dm))
    additional_loss = dm_loss

    nll_loss = tf.reduce_mean(lmb * tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logit))

    loss = nll_loss + additional_loss

    opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=mom)
    tvars = tf.trainable_variables()
    grads_and_vars = opt.compute_gradients(loss, tvars)
    train_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
    return loss, train_op, global_step


def build_eval_graph(x, y, x_ul):
    losses = {}
    logit = forward(x, is_training=False, update_batch_stats=False)
    nll_loss = L.ce_loss(logit, y)
    losses['NLL'] = nll_loss
    acc = L.accuracy(logit, y)
    losses['Acc'] = acc
    return losses


def main(_):
    numpy.random.seed(seed=FLAGS.seed)
    tf.set_random_seed(numpy.random.randint(1234))
    with tf.Graph().as_default() as g:
        with tf.device('/cpu:0'):
            lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
            mom = tf.placeholder(tf.float32, shape=[], name="momentum")

            x1 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 32, 32, 3], name="x1")
            x2 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 32, 32, 3], name="x2")
            y1 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 10], name="y1")

            x_eval = tf.placeholder(tf.float32, shape=[FLAGS.eval_batch_size, 32, 32, 3], name="x_eval")
            y_eval = tf.placeholder(tf.float32, shape=[FLAGS.eval_batch_size, 10], name="y_eval")
            x_uleval = tf.placeholder(tf.float32, shape=[FLAGS.eval_batch_size, 32, 32, 3], name="x_uleval")
            x_test = tf.placeholder(tf.float32, shape=[FLAGS.eval_batch_size, 32, 32, 3], name="x_test")
            y_test = tf.placeholder(tf.float32, shape=[FLAGS.eval_batch_size, 10], name="y_test")

            with tf.variable_scope("CNN") as scope:
                # Build training graph
                loss, train_op, global_step = build_training_graph(x1, y1, x2, lr, mom)
                scope.reuse_variables()
                # Build eval graph
                losses_eval_train = build_eval_graph(x_eval, y_eval, x_uleval)
                losses_eval_test = build_eval_graph(x_test, y_test, x_test)

            init_op = tf.global_variables_initializer()

            images, labels = inputs(batch_size=FLAGS.batch_size,
                                    train=True,
                                    validation=FLAGS.validation,
                                    shuffle=True,
                                    datadir = FLAGS.data_dir)

            ul_images = unlabeled_inputs(datadir = FLAGS.data_dir,
                                        batch_size=FLAGS.ul_batch_size,
                                         validation=FLAGS.validation,
                                         shuffle=True)

            images_eval_train, labels_eval_train = inputs(datadir = FLAGS.data_dir,
                                                            batch_size=FLAGS.eval_batch_size,
                                                          train=True,
                                                          validation=FLAGS.validation,
                                                          shuffle=True)
            ul_images_eval_train = unlabeled_inputs(datadir = FLAGS.data_dir,
                                                    batch_size=FLAGS.eval_batch_size,
                                                    validation=FLAGS.validation,
                                                    shuffle=True)

            images_eval_test, labels_eval_test = inputs(datadir = FLAGS.data_dir,
                                                        batch_size=FLAGS.eval_batch_size,
                                                        train=False,
                                                        validation=FLAGS.validation,
                                                        shuffle=True)

        if not FLAGS.logdir:
            logdir = None
            writer_train = None
            writer_test = None
        else:
            logdir = FLAGS.logdir
            writer_train = tf.summary.FileWriter(FLAGS.logdir + "/train", g)
            writer_test = tf.summary.FileWriter(FLAGS.logdir + "/test", g)

        saver = tf.train.Saver(tf.global_variables())
        sv = tf.train.Supervisor(
            is_chief=True,
            logdir=logdir,
            init_op=init_op,
            init_feed_dict={lr: FLAGS.learning_rate, mom: FLAGS.mom1},
            saver=saver,
            global_step=global_step,
            summary_op=None,
            summary_writer=None,
            save_model_secs=150, recovery_wait_secs=0)

        print("Training...")
        # config = tf.ConfigProto()
        # custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        # custom_op.name = "NpuOptimizer"
        # custom_op.parameter_map["use_off_line"].b = True
        # custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        # config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        config = tf.ConfigProto(allow_soft_placement=True)
        with sv.managed_session(config = npu_config_proto(config_proto = config)) as sess:

            for ep in range(FLAGS.num_epochs):
                if sv.should_stop():
                    break

                if ep < FLAGS.epoch_decay_start:
                    lr0 = FLAGS.learning_rate
                    mom0 = FLAGS.mom1
                else:
                    decayed_lr = ((FLAGS.num_epochs - ep) / float(
                        FLAGS.num_epochs - FLAGS.epoch_decay_start)) * FLAGS.learning_rate
                    lr0 = decayed_lr
                    mom0 = FLAGS.mom2
                sum_loss = 0
                start = time.time()
                for i in range(FLAGS.num_iter_per_epoch):
                    image, label, ulimage = sess.run([images, labels, ul_images])
                    # print(label)
                    #                     label = numpy.reshape(label,(FLAGS.batch_size,10))
                    label = numpy.reshape(label, (FLAGS.batch_size, 10))
                    _, batch_loss, _ = sess.run([train_op, loss, global_step],
                                                feed_dict={lr: lr0, mom: mom0, x1: image, y1: label, x2: ulimage})
                    sum_loss += batch_loss
                end = time.time()
                print("Epoch:", ep, "CE_loss_train:", sum_loss / FLAGS.num_iter_per_epoch, "elapsed_time:", end - start)
                # ft = open(filepath, 'a')
                # ft.write("Epoch: " + str(ep) + "CE_loss_train: " + str(
                #     sum_loss / FLAGS.num_iter_per_epoch) + "elapsed_time: " + str(end - start) + '\n')
                # ft.close()

                if (ep + 1) % FLAGS.eval_freq == 0 or ep + 1 == FLAGS.num_epochs:
                    # Eval on training data
                    act_values_dict = {}
                    for key, _ in losses_eval_train.items():
                        act_values_dict[key] = 0
                    n_iter_per_epoch = NUM_EVAL_EXAMPLES // FLAGS.eval_batch_size
                    for i in range(n_iter_per_epoch):
                        eval_image, eval_label, ul_eval_image = sess.run(
                            [images_eval_train, labels_eval_train, ul_images_eval_train])
                        eval_label = numpy.reshape(eval_label, (FLAGS.eval_batch_size, 10))
                        values = list(losses_eval_train.values())
                        act_values = sess.run(values, feed_dict={x_eval: eval_image, y_eval: eval_label,
                                                                 x_uleval: ul_eval_image})
                        for key, value in zip(list(act_values_dict.keys()), act_values):
                            act_values_dict[key] += value
                    summary = tf.Summary()
                    current_global_step = sess.run(global_step)
                    for key, value in act_values_dict.items():
                        print("train-" + key, value / n_iter_per_epoch)
                        # ft = open(filepath, 'a')
                        # ft.write("train-" + str(key) + ': ' + str(value / n_iter_per_epoch) + '\n')
                        # ft.close()
                        summary.value.add(tag=key, simple_value=value / n_iter_per_epoch)
                    if writer_train is not None:
                        writer_train.add_summary(summary, current_global_step)

                    # Eval on test data
                    act_values_dict = {}
                    for key, _ in losses_eval_test.items():
                        act_values_dict[key] = 0
                    n_iter_per_epoch = NUM_EVAL_EXAMPLES // FLAGS.eval_batch_size
                    for i in range(n_iter_per_epoch):
                        test_image, test_label = sess.run([images_eval_test, labels_eval_test])
                        test_label = numpy.reshape(test_label, (FLAGS.eval_batch_size, 10))
                        values = list(losses_eval_test.values())
                        act_values = sess.run(values, feed_dict={x_test: test_image, y_test: test_label})
                        for key, value in zip(list(act_values_dict.keys()), act_values):
                            act_values_dict[key] += value
                    summary = tf.Summary()
                    current_global_step = sess.run(global_step)
                    for key, value in act_values_dict.items():
                        print("test-" + key, value / n_iter_per_epoch)
                        # ft = open(filepath, 'a')
                        # ft.write("test-" + str(key) + ': ' + str(value / n_iter_per_epoch) + '\n')
                        # ft.close()
                        summary.value.add(tag=key, simple_value=value / n_iter_per_epoch)
                    if writer_test is not None:
                        writer_test.add_summary(summary, current_global_step)

            saver.save(sess, sv.save_path, global_step=global_step)
        sv.stop()
    #mox.file.copy_parallel(FLAGS.logdir, FLAGS.train_url)


if __name__ == "__main__":
   # os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = "0"
    #(npu_sess, npu_shutdown) = init_resource()
   # before()
    try:
        tf.app.run()
        after()
    finally:
        print('''download log''')

    #shutdown_resource(npu_sess, npu_shutdown)
    #close_session(npu_sess)
