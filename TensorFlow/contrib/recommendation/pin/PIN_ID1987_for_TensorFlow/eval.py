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


from __future__ import division
from __future__ import print_function
from sklearn.metrics import log_loss, roc_auc_score
from npu_bridge.npu_init import *

import json
import os
import sys
import time
from datetime import timedelta, datetime

import numpy as np
import tensorflow as tf1
import tensorflow.compat.v1 as tf

import __init__

from datasets import as_dataset
from print_hook import PrintHook
from models import as_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('worker_hosts', '10.58.14.147:12346,10.58.14.150:12347',
                           'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('worker_num_gpus', '2,2', 'Comma-separated list of integers')
tf.app.flags.DEFINE_string('job_name', '', 'One of ps, worker')
tf.app.flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
tf.app.flags.DEFINE_integer('num_ps', 1, 'Number of ps')
tf.app.flags.DEFINE_integer('num_workers', 2, 'Number of workers')
tf.app.flags.DEFINE_bool('distributed', False, 'Distributed training using parameter servers')
# tf.app.flags.DEFINE_bool('sync', False, 'Synchronized training')
tf.app.flags.DEFINE_integer('lazy_update', 1, 'Number of local steps by which variable update is delayed')

tf.app.flags.DEFINE_integer('num_shards', 1, 'Number of variable partitions')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of variable partitions')
tf.app.flags.DEFINE_bool('sparse_grad', False, 'Apply sparse gradient')

tf.app.flags.DEFINE_string('logdir', '../log', 'Directory for storing logs and models')
tf.app.flags.DEFINE_string('tag', '', 'Tag for logdir')
tf.app.flags.DEFINE_bool('restore', True, 'Restore from logdir')
tf.app.flags.DEFINE_bool('val', False, 'If True, use validation set, else use test set')
tf.app.flags.DEFINE_float('val_ratio', 0., 'Validation ratio')

tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer')
tf.app.flags.DEFINE_float('epsilon', 1e-4, 'Epsilon for adam')
tf.app.flags.DEFINE_float('init_val', 0.1, 'Initial accumulator value for adagrad')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
tf.app.flags.DEFINE_string('loss_mode', 'mean', 'Loss = mean, sum')

tf.app.flags.DEFINE_integer('batch_size', 2000, 'Training batch size')
tf.app.flags.DEFINE_integer('test_batch_size', 2000, 'Testing batch size')
tf.app.flags.DEFINE_string('dataset', 'ipinyou', 'Dataset = ipinyou, avazu, criteo, criteo_challenge')
tf.app.flags.DEFINE_string('model', 'pin', 'Model type = lr, fm, ffm, kfm, nfm, fnn, ccpm, deepfm, ipnn, kpnn, pin')

tf.app.flags.DEFINE_bool('input_norm', False, 'If true, the input will be normalized (set l2_norm to 1); else, do nothing')
tf.app.flags.DEFINE_bool('init_sparse', False, 'If true, embedding layer ~ uniform(0, sqrt(k)); else, embedding layer ~ xavier')
tf.app.flags.DEFINE_bool('init_fused', False, 'If true, sub-net weights ~ xavier([num_sub_net, in_node, out_node]); else xavier([in_node, out_node])')

tf.app.flags.DEFINE_integer('embed_size', 20, 'Embedding size')
tf.app.flags.DEFINE_string('nn_layers', '[' + '["full", 300],  ["act", "relu"], ' * 3 + '["full", 1]]',
                           'Network structure')
tf.app.flags.DEFINE_string('sub_nn_layers', '[["full", 40], ["ln", ""], ["act", "relu"], ["full", 5],  ["ln", ""]]',
                           'Sub-network structure')
tf.app.flags.DEFINE_float('l2_embed', 1e-6, 'L2 regularization')
tf.app.flags.DEFINE_float('l2_kernel', 1e-5, 'L2 regularization for kernels')
tf.app.flags.DEFINE_bool('wide', True, 'Wide term for pin')
tf.app.flags.DEFINE_bool('prod', True, 'Use product term as sub-net input')
tf.app.flags.DEFINE_string('kernel_type', 'vec', 'Kernel type = mat, vec, num')
tf.app.flags.DEFINE_bool('unit_kernel', False, 'Kernel in unit ball')
tf.app.flags.DEFINE_bool('fix_kernel', False, 'Fix kernel')

tf.app.flags.DEFINE_integer('num_rounds', 10, 'Number of training rounds')
tf.app.flags.DEFINE_integer('eval_level', 0, 'Evaluating frequency level')
tf.app.flags.DEFINE_float('decay', 1., 'Learning rate decay')
tf.app.flags.DEFINE_integer('log_frequency', 1000, 'Logging frequency')


# create log dir, e.g., ../log/data/model/timestamp/
def get_logdir(FLAGS):
    if FLAGS.restore:
        logdir = FLAGS.logdir
    else:
        tag = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S') if not FLAGS.distributed else ''
        logdir = '%s/%s/%s/%s' % (FLAGS.logdir, FLAGS.dataset, FLAGS.model, FLAGS.tag + tag)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfile = open(logdir + '/log', 'a')
    return logdir, logfile


# redirect stdout to log file
def redirect_stdout(logfile):
    def MyHookOut(text):
        logfile.write(text)
        logfile.flush()
        return 1, 0, text

    phOut = PrintHook()
    phOut.Start(MyHookOut)


def get_optimizer(opt, lr):
    opt = opt.lower()
    eps = FLAGS.epsilon
    init_val = FLAGS.init_val
    if opt == 'sgd' or opt == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif opt == 'adam':
        return tf.train.AdamOptimizer(learning_rate=lr, epsilon=eps)
    elif opt == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate=lr, initial_accumulator_value=init_val)


# sparse_add when using multiple gpus
def sparse_grads_mean(grads_and_vars):
    indices = []
    values = []
    dense_shape = grads_and_vars[0][0].dense_shape
    n = len(grads_and_vars)
    for g, _ in grads_and_vars:
        indices.append(g.indices)
        values.append(g.values / n)
    indices = tf.concat(indices, axis=0)
    values = tf.concat(values, axis=0)
    return tf.IndexedSlices(values=values, indices=indices, dense_shape=dense_shape)


# sync threads
def create_done_queue(i):
    with tf.device('/job:ps/task:%d' % (i)):
        return tf.FIFOQueue(FLAGS.num_workers, tf.int32, shared_name='done_queue' + str(i))


def create_done_queues():
    return [create_done_queue(i) for i in range(FLAGS.num_ps)]


def create_finish_queue(i):
    with tf.device('/job:worker/task:%d' % (i)):
        return tf.FIFOQueue(FLAGS.num_workers - 1, tf.int32, shared_name='done_queue' + str(i))


def create_finish_queues():
    return [create_finish_queue(0)]


class Trainer:
    def __init__(self):
        # parse params
        self.config = {}
        self.logdir, self.logfile = get_logdir(FLAGS=FLAGS)
        self.ckpt_dir = os.path.join(self.logdir, 'checkpoints')
        self.ckpt_name = 'model.ckpt'
        self.worker_dir = 'worker_%d' % FLAGS.task_index if FLAGS.distributed else ''
        self.sub_file = os.path.join(self.logdir, 'submission.%d.csv')
        redirect_stdout(self.logfile)

        if not FLAGS.distributed:
            self.num_gpus = 1
            self.total_num_gpus = self.num_gpus
        self.lazy_update = FLAGS.lazy_update
        self.train_data_param = {
            'gen_type': 'train',
            'random_sample': True,
            'batch_size': FLAGS.batch_size * self.num_gpus,
            'squeeze_output': False,
            'val_ratio': FLAGS.val_ratio,
        }
        self.valid_data_param = {
            'gen_type': 'valid' if FLAGS.val else 'test',
            'random_sample': False,
            'batch_size': FLAGS.test_batch_size * self.num_gpus,
            'squeeze_output': False,
            'val_ratio': FLAGS.val_ratio,
        }
        self.test_data_param = {
            'gen_type': 'test',
            'random_sample': False,
            'batch_size': FLAGS.test_batch_size * self.num_gpus,
            'squeeze_output': False,
        }
        self.train_logdir = os.path.join(self.logdir, 'train', self.worker_dir)
        self.valid_logdir = os.path.join(self.logdir, 'valid', self.worker_dir)
        self.test_logdir = os.path.join(self.logdir, 'test', self.worker_dir)
        self.gpu_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                         gpu_options={'allow_growth': True})

        # args setting
        self.model_param = {'l2_embed': FLAGS.l2_embed, 'num_shards': FLAGS.num_shards, 'input_norm': FLAGS.input_norm,
                            'init_sparse': FLAGS.init_sparse, 'init_fused': FLAGS.init_fused,
                            'loss_mode': FLAGS.loss_mode}

        if FLAGS.model != 'lr':
            self.model_param['embed_size'] = FLAGS.embed_size
        if FLAGS.model in ['fnn', 'ccpm', 'deepfm', 'ipnn', 'kpnn', 'pin']:
            self.model_param['nn_layers'] = [tuple(x) for x in json.loads(FLAGS.nn_layers)]
        if FLAGS.model in ['nfm', 'pin']:
            self.model_param['sub_nn_layers'] = [tuple(x) for x in json.loads(FLAGS.sub_nn_layers)]
        if FLAGS.model == 'pin':
            self.model_param['wide'] = FLAGS.wide
            self.model_param['prod'] = FLAGS.prod
        if FLAGS.model in {'kfm', 'kpnn'}:
            self.model_param['unit_kernel'] = FLAGS.unit_kernel
            self.model_param['fix_kernel'] = FLAGS.fix_kernel
            self.model_param['l2_kernel'] = FLAGS.l2_kernel
            self.model_param['kernel_type'] = FLAGS.kernel_type

        self.dump_config()

    def get_elapsed(self):
        return time.time() - self.start_time

    def get_timedelta(self, eta=None):
        eta = eta or (time.time() - self.start_time)
        return str(timedelta(seconds=eta))

    def dump_config(self):
        for k, v in getattr(FLAGS, '__flags').items():
            self.config[k] = getattr(FLAGS, k)
        for k, v in __init__.config.items():
            if k != 'default':
                self.config[k] = v
        self.config['train_data_param'] = self.train_data_param
        self.config['valid_data_param'] = self.valid_data_param
        self.config['test_data_param'] = self.test_data_param
        self.config['logdir'] = self.logdir
        config_json = json.dumps(self.config, indent=4, sort_keys=True, separators=(',', ':'))
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print(config_json)
        path_json = os.path.join(self.logdir, 'config.json')
        cnt = 1
        while os.path.exists(path_json):
            path_json = os.path.join(self.logdir, 'config%d.json' % cnt)
            cnt += 1
        print('Config json file:', path_json)
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        open(path_json, 'w').write(config_json)

    def build_graph(self):
        tf.reset_default_graph()
        self.dataset = as_dataset(FLAGS.dataset)

        with tf.device('/cpu:0'):
            with tf.variable_scope(tf.get_variable_scope()):
                self.global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=[],
                                                   initializer=tf.constant_initializer(0), trainable=False)
                self.learning_rate = tf.get_variable(name='learning_rate', dtype=tf.float32, shape=[],
                                                     initializer=tf.constant_initializer(
                                                         FLAGS.learning_rate),
                                                     trainable=False)
                # self.lr_decay_op = tf.assign(self.learning_rate, self.learning_rate * FLAGS.decay)
                self.opt = get_optimizer(FLAGS.optimizer, self.learning_rate)
                self.model = as_model(FLAGS.model, batch_size=FLAGS.batch_size,input_dim=self.dataset.num_features,
                                      num_fields=self.dataset.num_fields,
                                      **self.model_param)
                tf.get_variable_scope().reuse_variables()
                self.grads = self.opt.compute_gradients(self.model.loss)

        with tf.device('/cpu:0'):
            if self.lazy_update > 1:
                local_grads = []
                accumulate_op = []
                reset_op = []
                self.local_grads = []
                for grad, v in self.grads:
                    zero_grad = tf.zeros_like(v)
                    local_grad = tf.Variable(zero_grad, dtype=tf.float32, trainable=False,
                                             name=v.name.split(':')[0] + '_local_grad',
                                             collections=[tf.GraphKeys.LOCAL_VARIABLES])
                    self.local_grads.append(local_grad)
                    reset_grad = local_grad.assign(zero_grad)
                    if FLAGS.sparse_grad and isinstance(grad, tf.IndexedSlices):
                        accumulate_grad = local_grad.scatter_sub(-grad)
                    else:
                        accumulate_grad = local_grad.assign_add(grad)
                    local_grads.append((local_grad, v))
                    accumulate_op.append(accumulate_grad)
                    reset_op.append(reset_grad)
            if self.lazy_update > 1:
                self.update_op = self.opt.apply_gradients(local_grads, global_step=self.global_step)
                self.accumulate_op = tf.group(*accumulate_op)
                self.reset_op = tf.group(*reset_op)
            else:
                self.train_op = self.opt.minimize(self.model.loss, global_step=self.global_step)
            self.saver = tf.train.Saver()

    def sess_op(self):
        # 昇腾迁移
        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭remap
        return tf.Session(config=config)

    def get_nake_sess(self):
        sess = self.sess
        while type(sess).__name__ != 'Session':
            sess = sess._sess
        return sess

    def evaluate_batch(self, batch_xs, batch_ys):
        if self.num_gpus == 1:
            feed_dict = {self.model.inputs: batch_xs, self.model.labels: batch_ys}
            if self.model.training is not None:
                feed_dict[self.model.training] = False
            _preds_ = self.sess.run(fetches=self.model.preds, feed_dict=feed_dict)
            batch_preds = [_preds_.flatten()]
        else:
            fetches = []
            feed_dict = {}
            _batch = int(len(batch_ys) / self.num_gpus)
            _split = [_batch * i for i in range(1, self.num_gpus)]
            batch_xs = np.split(batch_xs, _split)
            batch_ys = np.split(batch_ys, _split)
            for i, model in enumerate(self.models):
                xs, ys = batch_xs[i], batch_ys[i]
                fetches.append(model.preds)
                feed_dict[model.inputs] = xs
                feed_dict[model.labels] = ys
                if model.training is not None:
                    feed_dict[model.training] = False
            _preds_ = self.sess.run(fetches=fetches, feed_dict=feed_dict)
            batch_preds = [x.flatten() for x in _preds_]
        del feed_dict
        return batch_preds

    def evaluate(self, gen, writer=None, eps=1e-6, submission=0):
        labels = []
        preds = []
        start_time = time.time()
        for batch_xs, batch_ys in gen:
            if len(batch_ys) < self.num_gpus:
                break
            if len(batch_xs) < FLAGS.batch_size:
                continue
            labels.append(batch_ys.flatten())
            preds.extend(self.evaluate_batch(batch_xs, batch_ys))
        labels = np.hstack(labels)
        preds = np.hstack(preds)
        _min_ = len(np.where(preds < eps)[0])
        _max_ = len(np.where(preds > 1 - eps)[0])
        print('%d samples are evaluated' % len(labels))
        if _min_ + _max_ > 0:
            print('EPS: %g, %d (%.2f) < eps, %d (%.2f) > 1-eps, %d (%.2f) are truncated' %
                  (eps, _min_, _min_ / len(preds), _max_, _max_ / len(preds), _min_ + _max_,
                   (_min_ + _max_) / len(preds)))
        preds[preds < eps] = eps
        preds[preds > 1 - eps] = 1 - eps
        if not submission:
            _log_loss_ = log_loss(y_true=labels, y_pred=preds)
            _auc_ = roc_auc_score(y_true=labels, y_score=preds)
            print('%s-Loss: %.6f, AUC: %.6f, Elapsed: %s' %
                  (gen.gen_type.capitalize(), _log_loss_, _auc_, str(timedelta(seconds=(time.time() - start_time)))))
            return _log_loss_, _auc_
        else:
            with open(self.sub_file % submission, 'w') as f:
                f.write('Id,Predicted\n')
                for i, p in enumerate(preds):
                    f.write('{0},{1}\n'.format(i + 60000000, p))
            print('Submission file: %s' % (self.sub_file % submission))

    def eval_func(self):
        with self.sess_op() as self.sess:

            self.test_gen = self.dataset.batch_generator(self.test_data_param)
            self.test_writer = tf.summary.FileWriter(logdir=self.test_logdir, graph=self.sess.graph, flush_secs=30)

            # TODO check restore
            checkpoint_state = tf.train.get_checkpoint_state(self.ckpt_dir)
            if checkpoint_state and checkpoint_state.model_checkpoint_path:
                self.saver.restore(self.sess, checkpoint_state.model_checkpoint_path)
                print('Restore model from:', checkpoint_state.model_checkpoint_path)
                print('Run initial evaluation...')
                self.evaluate(self.test_gen, self.test_writer)
            else:
                print('Restore failed')


def main(_):
    trainer = Trainer()
    if trainer.num_gpus == 1:
        trainer.build_graph()
    trainer.eval_func()


if __name__ == '__main__':
    tf.app.run()
