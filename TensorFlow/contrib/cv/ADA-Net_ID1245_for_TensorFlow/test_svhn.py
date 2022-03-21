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

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('device', '/gpu:0', "device")

tf.app.flags.DEFINE_string('dataset', 'cifar10', "{cifar10, svhn}")

tf.app.flags.DEFINE_string('logdir', "/svhn_model/", "logdir")
tf.app.flags.DEFINE_bool('validation', False, "")

tf.app.flags.DEFINE_integer('finetune_batch_size', 100, "the number of examples in a batch")
tf.app.flags.DEFINE_integer('finetune_iter', 100, "the number of iteration for finetuning of BN stats")
tf.app.flags.DEFINE_integer('eval_batch_size', 500, "the number of examples in a batch")


from svhn import inputs, unlabeled_inputs

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


def build_finetune_graph(x):
    logit = forward(x, is_training=True, update_batch_stats=True)
    with tf.control_dependencies([logit]):
        finetune_op = tf.no_op()
    return finetune_op


def build_eval_graph(x, y):
    logit = forward(x, is_training=False, update_batch_stats=False)
    n_corrects = tf.cast(tf.equal(tf.argmax(logit, 1), tf.argmax(y,1)), tf.int32)
    return tf.reduce_sum(n_corrects), tf.shape(n_corrects)[0] 


def main(_):
    with tf.Graph().as_default() as g:
        with tf.device('/cpu:0'):
            images_eval_train, _ = inputs(datadir = FLAGS.data_dir,batch_size=FLAGS.finetune_batch_size,
                                          validation=FLAGS.validation,
                                          shuffle=True)
            images_eval_test, labels_eval_test = inputs(datadir = FLAGS.data_dir,batch_size=FLAGS.eval_batch_size,
                                                        train=False,
                                                        validation=FLAGS.validation,
                                                        shuffle=False, num_epochs=1)

        with tf.device('/cpu:0'):
            x_eval = tf.placeholder(tf.float32, shape=[FLAGS.finetune_batch_size, 32, 32, 3], name="x_eval")
            x_test = tf.placeholder(tf.float32, shape=[FLAGS.eval_batch_size, 32, 32, 3], name="x_test")
            y_test = tf.placeholder(tf.float32, shape=[FLAGS.eval_batch_size, 10], name="y_test")
            with tf.variable_scope("CNN") as scope:
                # Build graph of finetuning BN stats
                finetune_op = build_finetune_graph(x_eval)
                scope.reuse_variables()
                # Build eval graph
                n_correct, m = build_eval_graph(x_test, y_test)

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session(config=npu_config_proto())
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(FLAGS.logdir)
        print("Checkpoints:", ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(tf.local_variables_initializer()) 
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)
        print("Finetuning...")
        for _ in range(FLAGS.finetune_iter):
            images_eval_train0 = sess.run(images_eval_train)
            sess.run(finetune_op,feed_dict = {x_eval:images_eval_train0})
            
        sum_correct_examples= 0
        sum_m = 0
        try:
            while not coord.should_stop():
                images_eval_test0, labels_eval_test0 = sess.run([images_eval_test, labels_eval_test])
                labels_eval_test0 = numpy.reshape(labels_eval_test0, (FLAGS.eval_batch_size, 10))
                _n_correct, _m = sess.run([n_correct, m],feed_dict = {x_test:images_eval_test0,y_test:labels_eval_test0})
                sum_correct_examples += _n_correct
                sum_m += _m
        except tf.errors.OutOfRangeError:
            print('Done evaluation -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        print("Test: num_test_examples:{}, num_correct_examples:{}, accuracy:{}".format(
              sum_m, sum_correct_examples, sum_correct_examples/float(sum_m)))
   
if __name__ == "__main__":
    tf.app.run()