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


from npu_bridge.npu_init import *
from npu_bridge.npu_init import *
import time

import numpy
import tensorflow as tf

import layers as L
import vat
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', "{cifar10}")
tf.app.flags.DEFINE_string('train_url', "../checkpoint", "train_url")
tf.app.flags.DEFINE_bool('validation', False, "")
tf.app.flags.DEFINE_integer('finetune_batch_size', 100, "the number of examples in a batch")
tf.app.flags.DEFINE_integer('finetune_iter', 100, "the number of iteration for finetuning of BN stats")
tf.app.flags.DEFINE_integer('eval_batch_size', 500, "the number of examples in a batch")
tf.app.flags.DEFINE_string('data_url', '../dataset/cifar10','where to store the dataset')


if FLAGS.dataset == 'cifar10':
    from cifar10 import inputs, unlabeled_inputs
else: 
    raise NotImplementedError


def build_finetune_graph(x):
    logit = vat.forward(x, is_training=True, update_batch_stats=True)
    with tf.control_dependencies([logit]):
        finetune_op = tf.no_op()
    return finetune_op


def build_eval_graph(x, y):
    logit = vat.forward(x, is_training=False, update_batch_stats=False)
    n_corrects = tf.cast(tf.equal(tf.argmax(logit, 1), tf.argmax(y,1)), tf.int32)
    return tf.reduce_sum(n_corrects), tf.shape(n_corrects)[0] 


def main(_):
    with tf.Graph().as_default() as g:
        images_eval_test, labels_eval_test = inputs(batch_size=FLAGS.eval_batch_size,
                                                    train=False,
                                                    validation=FLAGS.validation,
                                                    shuffle=False, num_epochs=1)

        with tf.variable_scope("CNN") as scope:
            # Build graph of finetuning BN stats
            finetune_op = build_finetune_graph(images_eval_train)
            scope.reuse_variables()
            # Build eval graph
            n_correct, m = build_eval_graph(images_eval_test, labels_eval_test)

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())

        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["mix_compile_mode"].b = True

        '''profiling'''
        #custom_op.parameter_map["profiling_mode"].b = True
        #custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/work/user-job-dir/profiling", "task_trace":"on" , "training_trace":"on"}')
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF

        sess = tf.Session(config=config)
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_url)
        print("Checkpoints:", ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        sess.run(tf.local_variables_initializer()) 
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)

        #print("Finetuning...")
        #for _ in range(FLAGS.finetune_iter):
        #    sess.run(finetune_op)
         
        sum_correct_examples= 0
        sum_m = 0
        try:
            while not coord.should_stop():
                _n_correct, _m = sess.run([n_correct, m])
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