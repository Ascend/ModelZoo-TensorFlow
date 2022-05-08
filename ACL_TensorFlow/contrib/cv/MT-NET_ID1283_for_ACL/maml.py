#
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
#
""" Code for the MAML algorithm and network definitions. """
import numpy as np

try:
    import special_grads
except KeyError as e:
    print ('WARNING: Cannot define MaxPoolGrad, likely already defined for this version of TensorFlow:', e)
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import mse, xent, conv_block, normalize

FLAGS = flags.FLAGS

## chip options
flags.DEFINE_string('chip', 'npu', "run on which chip, (npu or gpu or cpu)")
flags.DEFINE_string('platform', 'linux', 'runtime platform, linux or modelarts')
flags.DEFINE_string("obs_dir", '', 'obs result path, not need on gpu and apulis platform')
flags.DEFINE_boolean("profiling", False, "profiling for performance or not")

## Dataset/method options

flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('num_train_classes', -1, 'number of classes to train on (-1 for all).')
# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 40000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 1, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 1, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', .01, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
flags.DEFINE_integer('poly_order', 1, 'order of polynomial to generate')

## Model options
#flags.DEFINE_string('mod', '', 'modifications to original paper. None, split, both')
flags.DEFINE_bool('use_T', True, 'whether or not to use transformation matrix T')
flags.DEFINE_bool('use_M', True, 'whether or not to use mask M')
flags.DEFINE_bool('share_M', True, 'only effective if use_M is true, whether or not to '
                                    'share masks between weights'
                                    'that contribute to the same activation')
flags.DEFINE_float('temp', 1, 'temperature for gumbel-softmax')
flags.DEFINE_float('logit_init', 0, 'initial logit')
flags.DEFINE_string('norm', 'None', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('dim_hidden', 40, 'dimension of fc layer')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- use 32 for '
                                        'miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', True, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'logs/omniglot20way', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('debug', False, 'debug mode. uses less data for fast evaluation.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', False, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)') # 0.1 for omniglot


class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_num_updates=5):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.test_num_updates = test_num_updates
        if FLAGS.datasource in ['sinusoid', 'polynomial']:
            self.dim_hidden = [FLAGS.dim_hidden, FLAGS.dim_hidden]
            if FLAGS.use_T:
                self.forward = self.forward_fc_withT
            else:
                self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
            self.loss_func = mse
        elif FLAGS.datasource == 'omniglot' or FLAGS.datasource == 'miniimagenet':
            self.loss_func = xent
            self.classification = True
            if FLAGS.conv:
                self.dim_hidden = FLAGS.num_filters
                if FLAGS.use_T:
                    self.forward = self.forward_conv_withT
                else:
                    self.forward = self.forward_conv
                self.construct_weights = self.construct_conv_weights
            else:
                self.dim_hidden = [256, 128, 64, 64]
                self.forward = self.forward_fc
                self.construct_weights = self.construct_fc_weights
            if FLAGS.datasource == 'miniimagenet':
                self.channels = 3
            else:
                self.channels = 1
            self.img_size = int(np.sqrt(self.dim_input / self.channels))
        else:
            raise ValueError('Unrecognized data source.')

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        self.inputa = input_tensors['inputa']
        self.inputb = input_tensors['inputb']
        self.labela = input_tensors['labela']
        self.labelb = input_tensors['labelb']

        with tf.variable_scope('model', reuse=None) as training_scope:
            self.dropout_probs = {}
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]] * num_updates
            lossesb = [[]] * num_updates
            accuraciesb = [[]] * num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []
                mse_lossesb = []

                if self.classification:
                    task_accuraciesb = []

                train_keys = list(weights.keys())
                if FLAGS.use_M and FLAGS.share_M:
                    def make_shared_mask(key):
                        temperature = FLAGS.temp
                        logits = weights[key+'_prob']
                        logits = tf.stack([logits, tf.zeros(logits.shape)], 1)
                        U = tf.random_uniform(logits.shape, minval=0, maxval=1)
                        gumbel = -tf.log(-tf.log(U + 1e-20) + 1e-20)
                        y = logits + gumbel
                        gumbel_softmax = tf.nn.softmax(y / temperature)
                        gumbel_hard = tf.cast(tf.equal(gumbel_softmax, tf.reduce_max(gumbel_softmax, 1, keep_dims=True)), tf.float32)
                        mask = tf.stop_gradient(gumbel_hard - gumbel_softmax) + gumbel_softmax
                        return mask[:, 0]

                    def get_mask(masks, name):
                        mask = masks[[k for k in masks.keys() if name[-1] in k][0]]
                        if 'conv' in name:  # Conv
                            mask = tf.reshape(mask, [1, 1, 1, -1])
                            tile_size = weights[name].shape.as_list()[:3] + [1]
                            mask = tf.tile(mask, tile_size)
                        elif 'w' in name:  # FC
                            mask = tf.reshape(mask, [1, -1])
                            tile_size = weights[name].shape.as_list()[:1] + [1]
                            mask = tf.tile(mask, tile_size)
                        elif 'b' in name:  # Bias
                            mask = tf.reshape(mask, [-1])
                        return mask
                    if self.classification:
                        masks = {k: make_shared_mask(k) for k in ['conv1', 'conv2', 'conv3', 'conv4', 'w5']}
                    else:
                        masks = {k: make_shared_mask(k) for k in ['w1', 'w2', 'w3']}

                if FLAGS.use_M and not FLAGS.share_M:
                    def get_mask_noshare(key):
                        temperature = FLAGS.temp
                        logits = weights[key + '_prob']
                        logits = tf.stack([logits, tf.zeros(logits.shape)], 1)
                        U = tf.random_uniform(logits.shape, minval=0, maxval=1)
                        gumbel = -tf.log(-tf.log(U + 1e-20) + 1e-20)
                        y = logits + gumbel
                        gumbel_softmax = tf.nn.softmax(y / temperature)
                        gumbel_hard = tf.cast(tf.equal(gumbel_softmax, tf.reduce_max(gumbel_softmax, 1, keep_dims=True)), tf.float32)
                        out = tf.stop_gradient(gumbel_hard - gumbel_softmax) + gumbel_softmax
                        return tf.reshape(out[:, 0], weights[key].shape)

                train_keys = [k for k in weights.keys() if 'prob' not in k and 'f' not in k]
                train_weights = [weights[k] for k in train_keys]
                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter
                self.task_outputa = task_outputa
                task_lossa = self.loss_func(task_outputa, labela)
                grads = tf.gradients(task_lossa, train_weights)
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(train_keys, grads))

                fast_weights = dict(zip(weights.keys(), [weights[key] for key in weights.keys()]))

                def compute_weights(key):
                    prev_weights = fast_weights[key]
                    if key not in train_keys:
                        return prev_weights
                    if FLAGS.use_M and FLAGS.share_M:
                        mask = get_mask(masks, key)
                        new_weights = prev_weights - self.update_lr * mask * gradients[key]
                    elif FLAGS.use_M and not FLAGS.share_M:
                        mask = get_mask_noshare(key)
                        new_weights = prev_weights - self.update_lr * mask * gradients[key]
                    else:
                        new_weights = prev_weights - self.update_lr * gradients[key]
                    return new_weights

                fast_weights = dict(zip(
                    weights.keys(), [compute_weights(key) for key in weights.keys()]))

                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                loss = self.loss_func(output, labelb)
                task_lossesb.append(loss)

                for j in range(num_updates - 1):
                    output = self.forward(inputa, fast_weights, reuse=True)
                    loss = self.loss_func(output, labela)
                    train_weights = [fast_weights[k] for k in train_keys]
                    grads = tf.gradients(loss, train_weights)
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(train_keys, grads))

                    fast_weights = dict(zip(
                        weights.keys(), [compute_weights(key) for key in weights.keys()]))

                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    loss = self.loss_func(output, labelb)
                    task_lossesb.append(loss)

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                if self.classification:
                    task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1),
                                                                 tf.argmax(labela, 1))
                    for j in range(num_updates):
                        task_accuraciesb.append(
                            tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1),
                                                        tf.argmax(labelb, 1)))
                    task_output.extend([task_accuracya, task_accuraciesb])

                return task_output

            out_dtype = [tf.float32, [tf.float32] * num_updates, tf.float32, [tf.float32] * num_updates]
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32] * num_updates])

            if FLAGS.chip == 'npu':
                if self.classification:
                    outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = [], [], [], [], [], []
                    for i in range(FLAGS.meta_batch_size):
                        each_input = self.inputa[i], self.inputb[i], self.labela[i], self.labelb[i]
                        each_outputas, each_outputbs, each_lossesa, each_lossesb, each_accuraciesa, each_accuraciesb = task_metalearn(
                            each_input)
                        outputas.append(each_outputas)
                        outputbs.append(each_outputbs)
                        lossesa.append(each_lossesa)
                        lossesb.append(each_lossesb)
                        accuraciesa.append(each_accuraciesa)
                        accuraciesb.append(each_accuraciesb)
                    outputas = tf.stack(outputas)
                    outputbs = tf.unstack(tf.stack(outputbs), axis=1)
                    lossesa = tf.stack(lossesa)
                    lossesb = tf.unstack(tf.stack(lossesb), axis=1)
                    accuraciesa = tf.stack(accuraciesa)
                    accuraciesb = tf.unstack(tf.stack(accuraciesb), axis=1)
                else:
                    outputas, outputbs, lossesa, lossesb = [], [], [], []
                    for i in range(FLAGS.meta_batch_size):
                        each_input = self.inputa[i], self.inputb[i], self.labela[i], self.labelb[i]
                        each_outputas, each_outputbs, each_lossesa, each_lossesb = task_metalearn(
                            each_input)
                        outputas.append(each_outputas)
                        outputbs.append(each_outputbs)
                        lossesa.append(each_lossesa)
                        lossesb.append(each_lossesb)
                    outputas = tf.stack(outputas)
                    tmp = []
                    for i in outputbs:
                        for j in i:
                            outputbs = tmp.append(i)
                    lossesa = tf.stack(lossesa)
                    lossesb = tf.unstack(tf.stack(lossesb), axis=1)

        logit_keys = sorted([k for k in weights.keys() if 'prob' in k])
        logit_weights = [-weights[k] for k in logit_keys]
        probs = [tf.exp(w) / (1 + tf.exp(w)) for w in logit_weights]
        self.total_probs = [tf.reduce_mean(p) for p in probs]

        ## Performance & Optimization
        self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
        self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size)
                                                      for j in range(num_updates)]
        if self.classification:
            self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(
                FLAGS.meta_batch_size)
            self.metaval_total_accuracies2 = total_accuracies2 = [
                 tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]

        ## Summaries
        tf.summary.scalar(prefix + 'change probs', tf.reduce_mean(self.total_probs))
        tf.summary.scalar(prefix + 'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix + 'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix + 'Post-update loss, step ' + str(j + 1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix + 'Post-update accuracy, step ' + str(j + 1), total_accuracies2[j])

        for k, v in weights.items():
            tf.summary.histogram(k, v)
            if 'prob' in k:
                tf.summary.histogram('prob_'+k, tf.nn.softmax(tf.stack([v, tf.zeros(v.shape)], 1))[:, 0])

    ### Network construction functions (fc networks and conv networks)
    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1, len(self.dim_hidden)):
            weights['w' + str(i + 1)] = tf.Variable(
                tf.truncated_normal([self.dim_hidden[i - 1], self.dim_hidden[i]], stddev=0.01))
            weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w' + str(len(self.dim_hidden) + 1)] = tf.Variable(
            tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b' + str(len(self.dim_hidden) + 1)] = tf.Variable(tf.zeros([self.dim_output]))

        if FLAGS.use_M and not FLAGS.share_M:
            weights['w1_prob'] = tf.Variable(tf.truncated_normal([self.dim_input * self.dim_hidden[0]], stddev=.1))
            weights['b1_prob'] = tf.Variable(tf.truncated_normal([self.dim_hidden[0]], stddev=.1))
            for i in range(1, len(self.dim_hidden)):
                weights['w' + str(i + 1) + '_prob'] = tf.Variable(
                    tf.truncated_normal([self.dim_hidden[i - 1] * self.dim_hidden[i]], stddev=.1))
                weights['b' + str(i + 1) + '_prob'] = tf.Variable(
                    tf.truncated_normal([self.dim_hidden[i]], stddev=.1))
            weights['w' + str(len(self.dim_hidden) + 1) + '_prob'] = tf.Variable(
                tf.truncated_normal([self.dim_hidden[-1] * self.dim_output], stddev=0.1))
            weights['b' + str(len(self.dim_hidden) + 1) + '_prob'] = tf.Variable(
                tf.truncated_normal([self.dim_output], stddev=.1))
        elif FLAGS.use_M and FLAGS.share_M:
            weights['w1_prob'] = tf.Variable(FLAGS.logit_init * tf.ones([self.dim_hidden[0]]))
            for i in range(1, len(self.dim_hidden)):
                weights['w' + str(i + 1) + '_prob'] = tf.Variable(
                    FLAGS.logit_init * tf.ones([self.dim_hidden[i]]))
            weights['w' + str(len(self.dim_hidden) + 1) + '_prob'] = tf.Variable(
                FLAGS.logit_init * tf.ones([self.dim_output]))

        if FLAGS.use_T:
            weights['w1_f'] = tf.Variable(tf.eye(self.dim_hidden[0]))
            weights['w2_f'] = tf.Variable(tf.eye(self.dim_hidden[1]))
            weights['w3_f'] = tf.Variable(tf.eye(self.dim_output))
        return weights

    def forward_fc(self, inp, weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'],
                           activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1, len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w' + str(i + 1)]) + weights['b' + str(i + 1)],
                               activation=tf.nn.relu, reuse=reuse, scope=str(i + 1))
        return tf.matmul(hidden, weights['w' + str(len(self.dim_hidden) + 1)]) + \
               weights['b' + str(len(self.dim_hidden) + 1)]

    def forward_fc_withT(self, inp, weights, reuse=False):
        hidden = tf.matmul(tf.matmul(inp, weights['w1']) + weights['b1'], weights['w1_f'])
        hidden = normalize(hidden, activation=tf.nn.relu, reuse=reuse, scope='1')
        hidden = tf.matmul(tf.matmul(hidden, weights['w2']) + weights['b2'], weights['w2_f'])
        hidden = normalize(hidden, activation=tf.nn.relu, reuse=reuse, scope='2')
        hidden = tf.matmul(tf.matmul(hidden, weights['w3']) + weights['b3'], weights['w3_f'])
        return hidden

    def construct_conv_weights(self):
        weights = {}
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3
        channels = self.channels
        dim_hidden = self.dim_hidden

        def get_conv(name, shape):
            return tf.get_variable(name, shape, initializer=conv_initializer, dtype=dtype)

        def get_identity(dim, conv=True):
            return tf.Variable(tf.eye(dim, batch_shape=[1,1])) if conv \
                else tf.Variable(tf.eye(dim))

        weights['conv1'] = get_conv('conv1', [k, k, channels, self.dim_hidden])
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = get_conv('conv2', [k, k, dim_hidden, self.dim_hidden])
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = get_conv('conv3', [k, k, dim_hidden, self.dim_hidden])
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = get_conv('conv4', [k, k, dim_hidden, self.dim_hidden])
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        if FLAGS.datasource == 'miniimagenet':
            # assumes max pooling
            assert FLAGS.max_pool
            weights['w5'] = tf.get_variable('w5', [self.dim_hidden * 5 * 5, self.dim_output],
                                            initializer=fc_initializer)
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')

            if FLAGS.use_M and not FLAGS.share_M:
                weights['conv1_prob'] = tf.Variable(tf.truncated_normal([k * k * channels * self.dim_hidden], stddev=.01))
                weights['b1_prob'] = tf.Variable(tf.truncated_normal([self.dim_hidden], stddev=.01))
                weights['conv2_prob'] = tf.Variable(tf.truncated_normal([k * k * dim_hidden * self.dim_hidden], stddev=.01))
                weights['b2_prob'] = tf.Variable(tf.truncated_normal([self.dim_hidden], stddev=.01))
                weights['conv3_prob'] = tf.Variable(tf.truncated_normal([k * k * dim_hidden * self.dim_hidden], stddev=.01))
                weights['b3_prob'] = tf.Variable(tf.truncated_normal([self.dim_hidden], stddev=.01))
                weights['conv4_prob'] = tf.Variable(tf.truncated_normal([k * k * dim_hidden * self.dim_hidden], stddev=.01))
                weights['b4_prob'] = tf.Variable(tf.truncated_normal([self.dim_hidden], stddev=.01))
                weights['w5_prob'] = tf.Variable(tf.truncated_normal([dim_hidden *5*5* self.dim_output], stddev=.01))
                weights['b5_prob'] = tf.Variable(tf.truncated_normal([self.dim_output], stddev=.01))
            if FLAGS.use_M and FLAGS.share_M:
                weights['conv1_prob'] = tf.Variable(FLAGS.logit_init * tf.ones([self.dim_hidden]))
                weights['conv2_prob'] = tf.Variable(FLAGS.logit_init * tf.ones([self.dim_hidden]))
                weights['conv3_prob'] = tf.Variable(FLAGS.logit_init * tf.ones([self.dim_hidden]))
                weights['conv4_prob'] = tf.Variable(FLAGS.logit_init * tf.ones([self.dim_hidden]))
                weights['w5_prob'] = tf.Variable(FLAGS.logit_init * tf.ones([self.dim_output]))

            if FLAGS.use_T:
                weights['conv1_f'] = get_identity(self.dim_hidden, conv=True)
                weights['conv2_f'] = get_identity(self.dim_hidden, conv=True)
                weights['conv3_f'] = get_identity(self.dim_hidden, conv=True)
                weights['conv4_f'] = get_identity(self.dim_hidden, conv=True)
                weights['w5_f'] = get_identity(self.dim_output, conv=False)
        else:
            weights['w5'] = tf.Variable(tf.random_normal([dim_hidden, self.dim_output]), name='w5')
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
            if FLAGS.use_M and not FLAGS.share_M:
                weights['conv1_prob'] = tf.Variable(tf.truncated_normal([k * k * channels * self.dim_hidden], stddev=.01))
                weights['conv2_prob'] = tf.Variable(tf.truncated_normal([k * k * dim_hidden * self.dim_hidden], stddev=.01))
                weights['conv3_prob'] = tf.Variable(tf.truncated_normal([k * k * dim_hidden * self.dim_hidden], stddev=.01))
                weights['conv4_prob'] = tf.Variable(tf.truncated_normal([k * k * dim_hidden * self.dim_hidden], stddev=.01))
                weights['w5_prob'] = tf.Variable(tf.truncated_normal([dim_hidden * self.dim_output], stddev=.01))
            if FLAGS.use_M and FLAGS.share_M:
                weights['conv1_prob'] = tf.Variable(FLAGS.logit_init * tf.ones([self.dim_hidden]))
                weights['conv2_prob'] = tf.Variable(FLAGS.logit_init * tf.ones([self.dim_hidden]))
                weights['conv3_prob'] = tf.Variable(FLAGS.logit_init * tf.ones([self.dim_hidden]))
                weights['conv4_prob'] = tf.Variable(FLAGS.logit_init * tf.ones([self.dim_hidden]))
                weights['w5_prob'] = tf.Variable(FLAGS.logit_init * tf.ones([self.dim_output]))

            if FLAGS.use_T:
                weights['conv1_f'] = get_identity(self.dim_hidden, conv=True)
                weights['conv2_f'] = get_identity(self.dim_hidden, conv=True)
                weights['conv3_f'] = get_identity(self.dim_hidden, conv=True)
                weights['conv4_f'] = get_identity(self.dim_hidden, conv=True)
                weights['w5_f'] = get_identity(self.dim_output, conv=False)
        return weights

    def forward_conv(self, inp, weights, reuse=False, scope=''):
        # reuse is for the normalization parameters.
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])
        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')

        if FLAGS.datasource == 'miniimagenet':
            # last hidden layer is 6x6x64-ish, reshape to a vector
            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        else:
            hidden4 = tf.reduce_mean(hidden4, [1, 2])
        return tf.matmul(hidden4, weights['w5']) + weights['b5']

    def forward_conv_withT(self, inp, weights, reuse=False, scope=''):
        # reuse is for the normalization parameters.
        def conv_tout(inp, cweight, bweight, rweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID',
                       residual=False):
            stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]
            if FLAGS.max_pool:
                conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
            else:
                conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
            conv_output = tf.nn.conv2d(conv_output, rweight, no_stride, 'SAME')
            normed = normalize(conv_output, activation, reuse, scope)
            if FLAGS.max_pool:
                normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
            return normed

        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])
        hidden1 = conv_tout(inp, weights['conv1'], weights['b1'], weights['conv1_f'], reuse, scope + '0')
        hidden2 = conv_tout(hidden1, weights['conv2'], weights['b2'], weights['conv2_f'], reuse, scope + '1')
        hidden3 = conv_tout(hidden2, weights['conv3'], weights['b3'], weights['conv3_f'], reuse, scope + '2')
        hidden4 = conv_tout(hidden3, weights['conv4'], weights['b4'], weights['conv4_f'], reuse, scope + '3')

        if FLAGS.datasource == 'miniimagenet':
            # last hidden layer is 6x6x64-ish, reshape to a vector
            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        else:
            hidden4 = tf.reduce_mean(hidden4, [1, 2])
        hidden5 = tf.matmul(hidden4, weights['w5']) + weights['b5']
        return tf.matmul(hidden5, weights['w5_f'])