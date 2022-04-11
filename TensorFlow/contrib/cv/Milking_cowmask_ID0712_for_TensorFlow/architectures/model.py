# coding=utf-8
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

"""Wide Resnet Model with shake-shake regularization."""
from npu_bridge.npu_init import *
import logging
import time
import os
import numpy as np
import functools
import tensorflow as tf
import tensorflow.compat.v1 as tf
from architectures import shake

def create_stepped_lr(base_learning_rate, steps_per_epoch, lr_sched_steps):
    boundaries = [step[0] for step in lr_sched_steps]
    decays = [step[1] for step in lr_sched_steps]
    boundaries = np.array(boundaries) * steps_per_epoch
    boundaries = np.round(boundaries).astype(np.int32)
    values = np.array([1.0] + decays) * base_learning_rate
    return list(boundaries),list(values)

def confidence_thresholding(confidence, conf_thresh, conf_avg):
  """Confidence thresholding helper function.
  Args:
    confidence: per-sample confidence as a (batch_size,) array
    conf_thresh: confidence threshold
    conf_avg: if True, return the mean of the confidence masks
  Returns:
    (mask, conf_rate) tuple of the masks to use and the proportion of samples
      that pass the threshold
  """
  if conf_thresh > 0.0:
    conf_mask = (confidence > conf_thresh)
    conf_mask = tf.cast(conf_mask, tf.float32)
  else:
    conf_mask = np.ones_like(confidence)
  conf_mask_rate = tf.reduce_mean(conf_mask)
  if conf_avg:
    unsup_loss_mask = conf_mask_rate
  else:
    unsup_loss_mask = conf_mask
  return unsup_loss_mask, conf_mask_rate

def getter_ema(ema, getter, name, *args, **kwargs):
    """Exponential moving average getter for variable scopes.
    Args:
        ema: ExponentialMovingAverage object, where to get variable moving averages.
        getter: default variable scope getter.
        name: variable name.
        *args: extra args passed to default getter.
        **kwargs: extra args passed to default getter.

    Returns:
        If found the moving average variable, otherwise the default variable.
    """
    var = getter(name, *args, **kwargs)
    ema_var = ema.average(var)
    return ema_var if ema_var else var


class Model():
    def __init__(self,
                 batch_size,
                 num_epochs,
                 image_size,
                 n_classes,
                 train_ds,
                 eval_ds,
                 steps_per_epoch,
                 steps_per_eval,
                 base_learning_rate,
                 lr_sched_steps,
                 l2_reg,
                 ema,
                 sgd_momentum,
                 sgd_nesterov,
                 unsup_reg='none',
                 cons_weight=1.0,
                 conf_thresh=0.97,
                 conf_avg=False,
                 mix_reg='cowmix',
                 mix_logits=True,
                 mix_weight=0.0,
                 mix_conf_thresh=0.97,
                 mix_conf_avg=True,
                 mix_conf_mode='mix_conf',
                 mix_aug_separately=False,
                 ckpt=False,
                 model_path=None):
        self.holder = {}
        self.metrics = {}
        self.operas = {}
        self.eval_metrics = {}
        self.metrics['batch_size'] = batch_size
        self.metrics['num_epochs'] = num_epochs
        self.metrics['image_size'] = image_size
        self.metrics['n_classes'] = n_classes
        self.metrics['train_ds'] = train_ds
        self.metrics['eval_ds'] = eval_ds
        self.metrics['steps_per_epoch'] = steps_per_epoch
        self.metrics['steps_per_eval'] = steps_per_eval
        self.metrics['base_learning_rate'] = base_learning_rate
        self.metrics['lr_sched_steps'] = lr_sched_steps
        self.metrics['l2_reg'] = l2_reg
        self.metrics['ema'] = ema
        self.metrics['sgd_momentum'] = sgd_momentum
        self.metrics['sgd_nesterov'] = sgd_nesterov
        self.metrics['unsup_reg'] = unsup_reg
        self.metrics['cons_weight'] = cons_weight
        self.metrics['conf_thresh'] = conf_thresh
        self.metrics['conf_avg'] = conf_avg
        self.metrics['mix_reg'] = mix_reg
        self.metrics['mix_logits'] = mix_logits
        self.metrics['mix_weight'] = mix_weight
        self.metrics['mix_conf_thresh'] = mix_conf_thresh
        self.metrics['mix_conf_avg'] = mix_conf_avg
        self.metrics['mix_conf_mode'] = mix_conf_mode
        self.metrics['mix_aug_separately'] = mix_aug_separately
        self.metrics['model_path'] = model_path
        self.seed = 123
        self.create_net()
        self.saver = tf.train.Saver()
        config_proto = tf.ConfigProto()
        custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        config = npu_config_proto(config_proto=config_proto)
        self.sess = tf.Session(config=config)
        # self.sess = tf.Session(config=npu_config_proto())
        self.metrics['init'] = tf.global_variables_initializer()
        if ckpt:
            model_file = tf.train.latest_checkpoint(self.metrics['model_path'])
            self.saver.restore(self.sess, model_file)
        else:
            self.sess.run(self.metrics['init'])

    """Wide ResNet block with shake-shake."""
    def shakeShakeBlock(self, x, channels, strides=(1,1), train=True, name= None):
        a = b = residual = x
        a = tf.nn.relu(a)
        a = tf.layers.conv2d(a, channels, (3, 3), strides, padding='SAME',
                             kernel_initializer=tf.variance_scaling_initializer(), name=''.join([name, '_conv_a1']))
        with tf.compat.forward_compatibility_horizon(2019, 5, 1):
            a = tf.layers.batch_normalization(a, momentum=0.9, epsilon=1e-5, training=train, name=''.join([name, '_bn1']))
        a = tf.nn.relu(a)
        a = tf.layers.conv2d(a, channels, (3, 3), padding='SAME', kernel_initializer=tf.variance_scaling_initializer(),
                             name=''.join([name, '_conv_a2']))
        with tf.compat.forward_compatibility_horizon(2019, 5, 1):
            a = tf.layers.batch_normalization(a, momentum=0.9, epsilon=1e-5, training=train, name=''.join([name, '_bn2']))
        b = tf.nn.relu(b)
        b = tf.layers.conv2d(b, channels, (3, 3), strides, padding='SAME',
                             kernel_initializer=tf.variance_scaling_initializer(), name=''.join([name, '_conv_b1']))
        with tf.compat.forward_compatibility_horizon(2019, 5, 1):
            b = tf.layers.batch_normalization(b, momentum=0.9, epsilon=1e-5, training=train, name=''.join([name, '_bn3']))
        b = tf.nn.relu(b)
        b = tf.layers.conv2d(b, channels, (3, 3), padding='SAME', kernel_initializer=tf.variance_scaling_initializer(),
                             name=''.join([name, '_conv_b2']))
        with tf.compat.forward_compatibility_horizon(2019, 5, 1):
            b = tf.layers.batch_normalization(b, momentum=0.9, epsilon=1e-5, training=train, name=''.join([name, '_bn4']))

        if train:  # check
            ab = shake.shake_shake_train(a, b, seed=self.seed + 1, name=name)
        else:
            ab = shake.shake_shake_eval(a, b)
        # Apply an up projection in case of channel mismatch
        if (residual.get_shape()[-1] != channels) or strides != (1, 1):
            residual = tf.layers.conv2d(residual, channels, (3, 3), strides, padding='SAME',
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name=''.join([name, 'conv_res']))
            with tf.compat.forward_compatibility_horizon(2019, 5, 1):
                residual = tf.layers.batch_normalization(residual, momentum=0.9, epsilon=1e-5, training=train,
                                                     name=''.join([name, '_bn_res']))
        return residual + ab

    """Defines a WideResnetGroup."""
    def wideResnetShakeShakeGroup(self,
            x,
            blocks_per_group,
            channels,
            strides=(1, 1),
            train=True,
            name=None):
        for i in range(blocks_per_group):

            x = self.shakeShakeBlock(
                x,
                channels,
                strides if i == 0 else (1, 1),
                train=train,
                name=''.join([name,'_',str(i)]))
        return x

    """Defines the WideResnet Model."""
    def wideResnetShakeShake(self,
            x,
            num_outputs,
            blocks_per_group = 4,
            channel_multiplier = 6,
            getter = None,
            train=True):
        with tf.variable_scope('stu', reuse=tf.AUTO_REUSE, custom_getter=getter):
            x = tf.layers.conv2d(x, 16, (3,3), padding='SAME', kernel_initializer=tf.variance_scaling_initializer(), name='init_conv')
            with tf.name_scope('shake_g1'):
                x = self.wideResnetShakeShakeGroup(
                    x,
                    blocks_per_group,
                    16 * channel_multiplier,
                    train=train,
                    name= 'shake_g1')
            with tf.name_scope('shake_g2'):
                x = self.wideResnetShakeShakeGroup(
                    x,
                    blocks_per_group,
                    32 * channel_multiplier,
                    strides=(2, 2),
                    train=train,
                    name= 'shake_g2')
            with tf.name_scope('shake_g3'):
                x = self.wideResnetShakeShakeGroup(
                    x,
                    blocks_per_group,
                    64 * channel_multiplier, strides=(2, 2),
                    train=train,
                    name= 'shake_g3')
            x = tf.nn.relu(x)
            x = tf.layers.average_pooling2d(x, (8, 8), (1, 1), padding='VALID', name='pool')
            x = tf.reshape(x,(tf.shape(x)[0], 384))
            x = tf.layers.dense(x, num_outputs,  name='den')
            return x

    def create_net(self):
        # input_shape = (batch_size, image_size, image_size, 3)
        self.holder['sup_x'] = tf.placeholder(shape= [self.metrics['batch_size'], self.metrics['image_size'], self.metrics['image_size'], 3], dtype=tf.float32)
        self.holder['y'] = tf.placeholder(shape=[None], dtype=tf.int64, name='y')
        self.holder['unsup_x0'] = tf.placeholder(shape=[self.metrics['batch_size'], self.metrics['image_size'], self.metrics['image_size'], 3], dtype=tf.float32)
        self.holder['unsup_x1'] = tf.placeholder(shape=[self.metrics['batch_size'], self.metrics['image_size'], self.metrics['image_size'], 3], dtype=tf.float32)
        self.holder['val_x'] = tf.placeholder(shape=[None, self.metrics['image_size'], self.metrics['image_size'], 3], dtype=tf.float32, name='val_x')
        print("------------self.metrics['batch_size']------------",self.metrics['batch_size'])
        #create net
        self.operas['global_step'] = tf.Variable(0, trainable=False,name='global_step')
        self.operas['y_one_hot'] = tf.one_hot(tf.cast(self.holder['y'], dtype=tf.int32), self.metrics['n_classes'])
        # train operaters
        self.operas['sup_logits_stu'] = self.wideResnetShakeShake(self.holder['sup_x'], self.metrics['n_classes'], train=True)  # supervised learning forward propagation
        self.operas['ema'] = tf.train.ExponentialMovingAverage(decay=self.metrics['ema'], num_updates=self.operas['global_step'])
        stu_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='stu')
        self.operas['ema_op'] = self.operas['ema'].apply(stu_params)
        ema_getter = functools.partial(getter_ema, self.operas['ema'])
        if self.metrics['unsup_reg'] is not None:
            unsup_x1 = self.metrics['unsup_reg'].perturb_sample(self.holder['unsup_x1'], self.seed+10)
            unsup_logits_tea = self.wideResnetShakeShake(self.holder['unsup_x0'], self.metrics['n_classes'], getter=ema_getter, train=False)
            self.operas['unsup_logits_tea'] = tf.stop_gradient(unsup_logits_tea)
            self.operas['unsup_logits_stu'] = self.wideResnetShakeShake(unsup_x1, self.metrics['n_classes'],train=True)
        if self.metrics['mix_reg'] is not None:
            if self.metrics['mix_aug_separately']:
                x0_mix_tea = self.holder['unsup_x1']
            else:
                x0_mix_tea = self.holder['unsup_x0']
            x0_mix = self.holder['unsup_x0']
            x1_mix = tf.concat([x0_mix[1:, ...], x0_mix[:1, ...]], axis=0)
            x_mix, mix_blend_facs = self.metrics['mix_reg'].mix_images(x0_mix, x1_mix, self.seed+20)
            if self.metrics['unsup_reg'] is not None:
                logits0_mix_tea = self.operas['unsup_logits_tea']
            else:
                logits0_mix_tea = self.wideResnetShakeShake(x0_mix_tea, self.metrics['n_classes'], getter=ema_getter, train=False)
                logits0_mix_tea = tf.stop_gradient(logits0_mix_tea)
            self.operas['logits_mix_stu'] = self.wideResnetShakeShake(x_mix, self.metrics['n_classes'],train=True)
        self.operas['ops'] = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        '''val operaters'''
        self.operas['val_logits_stu'] = self.wideResnetShakeShake(self.holder['val_x'], self.metrics['n_classes'], train=False)
        self.operas['val_logits_tea'] = self.wideResnetShakeShake(self.holder['val_x'], self.metrics['n_classes'], getter= ema_getter, train=False)
        stu_logits, tea_logits, y_one_hot = self.operas['val_logits_stu'], self.operas['val_logits_tea'], self.operas[
            'y_one_hot']
        stu_accu_i = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(stu_logits, 1), self.holder['y']), tf.float32))
        tea_accu_i = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tea_logits, 1), self.holder['y']), tf.float32))
        stu_loss_i = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=stu_logits))
        tea_loss_i = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=tea_logits))
        self.operas['eval_metrics'] = [stu_accu_i, tea_accu_i, stu_loss_i, tea_loss_i]

        self.operas['sup_loss'] = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.operas['y_one_hot'], logits=self.operas['sup_logits_stu']))  # calculate loss
        loss = self.operas['sup_loss']
        self.operas['sup_auccracy'] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.operas['sup_logits_stu'], 1), self.holder['y']), tf.float32))

        '''cowout'''
        if self.metrics['unsup_reg'] is not None:
            # Logits -> probs
            unsup_prob_tea = tf.nn.softmax(self.operas['unsup_logits_tea'])
            unsup_prob_stu = tf.nn.softmax(self.operas['unsup_logits_stu'])
            # Confidence thresholding
            unsup_loss_mask, conf_mask_rate = confidence_thresholding(
                tf.reduce_max(unsup_prob_tea, axis=-1, keepdims=True), self.metrics['conf_thresh'], self.metrics['conf_avg'])
            # Unsupervised loss
            unsup_var_loss = ((unsup_prob_stu - unsup_prob_tea) ** 2) * unsup_loss_mask
            cons_loss = tf.reduce_mean(tf.reduce_sum(unsup_var_loss,axis=-1))
            loss = loss + cons_loss * self.metrics['cons_weight']
        else:
            cons_loss = tf.zeros(shape=(1,), dtype=tf.float32)
            conf_mask_rate = tf.zeros(shape=(1,), dtype=tf.float32)
            self.operas['unsup_logits_tea'] = None
            unsup_prob_tea = None
        ''''''

        '''cowmix'''
        if self.metrics['mix_reg'] is not None:
            # Logits -> probs, using temperature
            prob0_mix_tea = tf.nn.softmax(logits0_mix_tea)
            prob0_mix_tea_conf = tf.nn.softmax(logits0_mix_tea)

            logits1_mix_tea = tf.concat(
                [logits0_mix_tea[1:, ...], logits0_mix_tea[:1, ...]], axis=0)
            prob1_mix_tea = tf.concat(
                [prob0_mix_tea[1:, ...], prob0_mix_tea[:1, ...]], axis=0)
            prob1_mix_tea_conf = tf.concat(
                [prob0_mix_tea_conf[1:, ...], prob0_mix_tea_conf[:1, ...]], axis=0)

            # Apply mix
            if self.metrics['mix_logits']:
                logits_mix_tea = logits0_mix_tea + (logits1_mix_tea - logits0_mix_tea) * mix_blend_facs[:, None]
                prob_mix_tea = tf.nn.softmax(logits_mix_tea)
                prob_mix_tea_conf = tf.nn.softmax(logits_mix_tea)
            else:
                prob_mix_tea = prob0_mix_tea + (prob1_mix_tea - prob0_mix_tea) * mix_blend_facs[:, None]
                prob_mix_tea_conf = prob0_mix_tea_conf + \
                                    (prob1_mix_tea_conf - prob0_mix_tea_conf) * mix_blend_facs[:, None]

            prob_mix_stu = tf.nn.softmax(self.operas['logits_mix_stu'])

            if self.metrics['mix_conf_mode'] == 'mix_prob':
                conf_tea = tf.reduce_max(prob_mix_tea_conf, axis=-1, keepdims=True)
            elif self.metrics['mix_conf_mode'] == 'mix_conf':
                conf0_tea = tf.reduce_max(prob0_mix_tea_conf, axis=-1, keepdims=True)
                conf1_tea = tf.reduce_max(prob1_mix_tea_conf, axis=-1, keepdims=True)
                conf_tea = conf0_tea + (conf1_tea - conf0_tea) * mix_blend_facs[:, None]
            else:
                raise RuntimeError

            # Confidence thresholding
            mix_loss_mask, mix_conf_mask_rate = confidence_thresholding(conf_tea, self.metrics['mix_conf_thresh'], self.metrics['mix_conf_avg'])

            # Mix loss
            mix_var_loss = ((prob_mix_stu - prob_mix_tea) ** 2) * mix_loss_mask
            mix_loss = tf.reduce_mean(tf.reduce_sum(mix_var_loss,axis=-1))
            loss = loss + mix_loss * self.metrics['mix_weight']

        else:
            mix_loss = tf.zeros((1,), dtype=tf.float32)
            mix_conf_mask_rate = tf.zeros((1,), dtype=tf.float32)
        ''''''

        # l2-regularization
        if self.metrics['l2_reg'] > 0:
            weight_stu = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='stu')

            a = [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in weight_stu if v.shape.ndims>1]
            weight_penalty = self.metrics['l2_reg'] * tf.add_n(a)
            loss = loss + weight_penalty
        self.operas['loss'] = loss
        '''optimizer'''
        base_learning_rate = self.metrics['base_learning_rate'] * self.metrics['batch_size'] / 256.
        boundaries, values = create_stepped_lr(base_learning_rate, self.metrics['steps_per_epoch'], self.metrics['lr_sched_steps'])
        self.operas['learning_rate'] = tf.train.piecewise_constant_decay(self.operas['global_step'], boundaries=boundaries, values=values)
        loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=65536, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        opt_tmp = tf.train.MomentumOptimizer(self.operas['learning_rate'], momentum=self.metrics['sgd_momentum'], use_nesterov=self.metrics['sgd_nesterov'])
        opt = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)
        self.operas['opt'] = opt.minimize(self.operas['loss'], global_step=self.operas['global_step'])
        #self.operas['opt'] = tf.train.MomentumOptimizer(self.operas['learning_rate'], momentum=self.metrics['sgd_momentum'], use_nesterov=self.metrics['sgd_nesterov']).minimize(self.operas['loss'], global_step=self.operas['global_step'])
        self.operas['ops'].append(self.operas['opt'])
        self.operas['ops'].append(self.operas['ema_op'])
        with tf.control_dependencies(self.operas['ops']):
            self.operas['train_op'] = tf.no_op(name = 'train')

    def train_step(self,sup_x,sup_y,unsup_x0,unspu_x1):
        self.eval_metrics['train_loss'], a, = self.sess.run([self.operas['loss'], self.operas['train_op']],feed_dict={self.holder['sup_x']: sup_x,
                                                                                      self.holder['y']: sup_y,
                                                                                      self.holder['unsup_x0']: unsup_x0,
                                                                                      self.holder['unsup_x1']: unspu_x1})

    def compute_epoch_metrics(self):
        """Compute student_net  (loss and accuracy).
        """
        eval_iterator = tf.data.make_initializable_iterator(self.metrics['eval_ds'])
        self.sess.run(eval_iterator.initializer)
        next_eval_batch = eval_iterator.get_next()
        #stu metrics
        stu_accuracy = []
        stu_loss = []
        tea_accuracy = []
        tea_loss = []
        for i in range(self.metrics['steps_per_eval']):
            eval_batch = self.sess.run((next_eval_batch))
            eval_x = eval_batch['image']
            eval_y = eval_batch['label']
            stu_accu_i, tea_accu_i, stu_loss_i, tea_loss_i = self.sess.run(self.operas['eval_metrics'], feed_dict={self.holder['val_x']: eval_x, self.holder['y']: eval_y})
            stu_accuracy.append(stu_accu_i)
            tea_accuracy.append(tea_accu_i)
            stu_loss.append(stu_loss_i)
            tea_loss.append(tea_loss_i)
        self.eval_metrics['stu_loss'], self.eval_metrics['tea_loss'], self.eval_metrics['stu_accuracy'], self.eval_metrics['tea_accuracy'] = np.mean(stu_loss), np.mean(tea_loss), np.mean(stu_accuracy), np.mean(tea_accuracy)

    '''
    train loop
    '''
    def train(self):
        train_iterator = tf.data.make_initializable_iterator(self.metrics['train_ds'])
        next_train_batch = train_iterator.get_next()


        for epoch in range(self.metrics['num_epochs']):
            t1 = time.time()
            self.sess.run(train_iterator.initializer)
            for j in range(self.metrics['steps_per_epoch']):
                t_start = time.time()
                #print('step:',j)
                train_batch = self.sess.run(next_train_batch)
                sup_x = train_batch['sup_image']
                sup_y = train_batch['sup_label']
                unsup_x0 = train_batch['unsup_image0']
                unspu_x1 = train_batch['unsup_image1']
                self.train_step(sup_x, sup_y, unsup_x0, unspu_x1)
                t_end = time.time()
                perf = t_end - t_start
                fps = self.metrics['batch_size'] / perf    #(256)/       1
                print('epoch: {} step: {} perf: {:.5f} FPS: {:.5f} train_loss: {:.6f}'.format(epoch + 1, j, perf, fps, self.eval_metrics['train_loss']))
            self.saver.save(self.sess, self.metrics['model_path'] + 'milking_cowmask.ckpt', global_step=epoch + 1)

            self.compute_epoch_metrics()
            t2 = time.time()
            info = 'Epoch{} Time: {:.0f}  Stu_loss: {:.6f} accuracy: {:.3%} Tea: loss: {:.6f} accuracy: {:.3%}'.format(
                    epoch+1, t2-t1, self.eval_metrics['stu_loss'],
                    self.eval_metrics['stu_accuracy'], self.eval_metrics['tea_loss'],
                    self.eval_metrics['tea_accuracy'])
            print(info)
            #logging.info(info)
        tf.io.write_graph(self.sess.graph, self.metrics['model_path'], 'graph.pbtxt', as_text=True)
        self.sess.close()