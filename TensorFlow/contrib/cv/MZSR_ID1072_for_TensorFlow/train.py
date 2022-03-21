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
from utils import *
import model
import time
from config import *
from npu_bridge.estimator.npu.npu_loss_scale_optimizer import NPULossScaleOptimizer
from npu_bridge.estimator.npu.npu_loss_scale_manager import FixedLossScaleManager
from npu_bridge.estimator.npu.npu_loss_scale_manager import ExponentialUpdateLossScaleManager

class Train(object):
    def __init__(self, trial, step, size, scale_list, meta_batch_size, meta_lr, meta_iter, task_batch_size, task_lr, task_iter, data_generator, checkpoint_dir, conf):
        print('[*] Initialize Training')
        self.trial = trial
        self.step=step
        self.HEIGHT=size[0]
        self.WIDTH=size[1]
        self.CHANNEL=size[2]
        self.scale_list=scale_list

        self.META_BATCH_SIZE = meta_batch_size
        self.META_LR = meta_lr
        self.META_ITER = meta_iter

        self.TASK_BATCH_SIZE = task_batch_size
        self.TASK_LR = task_lr
        self.TASK_ITER = task_iter

        self.data_generator=data_generator
        self.checkpoint_dir=checkpoint_dir
        self.conf=conf

        '''placeholders'''
        self.inputa = tf.placeholder(dtype=tf.float32, shape=[self.META_BATCH_SIZE, self.TASK_BATCH_SIZE, self.HEIGHT, self.WIDTH, self.CHANNEL])
        self.inputb = tf.placeholder(dtype=tf.float32, shape=[self.META_BATCH_SIZE, self.TASK_BATCH_SIZE, self.HEIGHT, self.WIDTH, self.CHANNEL])

        self.labela = tf.placeholder(dtype=tf.float32, shape=[self.META_BATCH_SIZE, self.TASK_BATCH_SIZE, self.HEIGHT, self.WIDTH, self.CHANNEL])
        self.labelb = tf.placeholder(dtype=tf.float32, shape=[self.META_BATCH_SIZE, self.TASK_BATCH_SIZE, self.HEIGHT, self.WIDTH, self.CHANNEL])

        '''model'''
        self.PARAM=model.Weights(scope='MODEL')
        self.weights=self.PARAM.weights

        self.MODEL = model.MODEL(name='MODEL')

    def construct_model(self):
        self.stop_grad=tf.Variable(True, name='stop_grad', trainable=False)

        def task_metalearn(inp):
            inputa, inputb, labela, labelb = inp
            loss_func = tf.losses.absolute_difference

            task_outputbs, task_lossesb = [], []

            self.MODEL.forward(inputa, self.weights)
            task_outputa = self.MODEL.output

            weights = self.MODEL.param
            task_lossa = loss_func(labela, task_outputa)

            grads = tf.gradients(task_lossa, list(weights.values()))
            grads = tf.cond(self.stop_grad, lambda: [tf.stop_gradient(grad) for grad in grads], lambda: grads)

            gradients = dict(zip(weights.keys(), grads))
            fast_weights = dict(
                zip(weights.keys(), [weights[key] - self.TASK_LR * gradients[key] for key in weights.keys()]))

            self.MODEL.forward(inputb, fast_weights)
            output = self.MODEL.output

            task_outputbs.append(output)
            task_lossesb.append(loss_func(labelb, output))

            for j in range(self.TASK_ITER - 1):
                self.MODEL.forward(inputa, fast_weights)
                output_s = self.MODEL.output

                loss = loss_func(labela, output_s)
                grads = tf.gradients(loss, list(fast_weights.values()))

                grads = tf.cond(self.stop_grad, lambda: [tf.stop_gradient(grad) for grad in grads], lambda: grads)

                gradients = dict(zip(fast_weights.keys(), grads))
                fast_weights = dict(zip(fast_weights.keys(),
                                        [fast_weights[key] - self.TASK_LR* gradients[key] for key in
                                         fast_weights.keys()]))

                self.MODEL.forward(inputb, fast_weights)
                output=self.MODEL.output

                task_outputbs.append(output)
                task_lossesb.append(loss_func(labelb, output))

            task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

            return task_output


        out_dtype = [tf.float32, [tf.float32] * self.TASK_ITER, tf.float32, [tf.float32] * self.TASK_ITER]

        outputas, outputbs, lossesa, lossesb = [], [], [], []
        for i in range(self.META_BATCH_SIZE):
            each_input = self.inputa[i], self.inputb[i], self.labela[i], self.labelb[i]
            each_outputas, each_outputbs, each_lossesa, each_lossesb = task_metalearn(each_input)
            outputas.append(each_outputas)
            outputbs.append(each_outputbs)
            lossesa.append(each_lossesa)
            lossesb.append(each_lossesb)
        self.outputas = tf.stack(outputas)
        self.outputbs = tf.unstack(tf.stack(outputbs), axis=1)
        self.lossesa = tf.stack(lossesa)
        self.lossesb = tf.unstack(tf.stack(lossesb), axis=1)

        #result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype,
        #                   parallel_iterations=self.META_BATCH_SIZE)

        #self.outputas, self.outputbs, self.lossesa, self.lossesb = result

    def __call__(self):
        PRINT_ITER=100
        SAVE_ITER=2500
        SECOND_ORDER_GRAD_ITER=0 # For the 1st-order approximation. Until this step, 1st-order approximation is used for fast training

        print('[*] Setting Train Configuration')

        self.construct_model()

        self.global_step=tf.Variable(self.step, name='global_step', trainable=False)
        self.second_grad_on=tf.assign(self.stop_grad, False)

        '''losses'''
        self.total_loss1 = tf.reduce_sum(self.lossesa) / tf.to_float(self.META_BATCH_SIZE)
        self.total_losses2 =  [tf.reduce_sum(self.lossesb[j]) / tf.to_float(self.META_BATCH_SIZE) for j in range(self.TASK_ITER)]

        '''weighted loss'''
        self.LW=self.get_loss_weights()
        self.weighted_total_losses2 = tf.reduce_mean(tf.multiply(tf.convert_to_tensor(self.total_losses2),self.LW))
        # self.weighted_total_losses2=self.total_losses2[-1]

        '''Optimizers'''
        self.pretrain_op = tf.train.AdamOptimizer(self.META_LR).minimize(self.total_loss1)


        # self.gvs = self.opt.compute_gradients(self.total_losses2[self.META_BATCH_SIZE-1])
        ##loss scale
        loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
                                                               decr_every_n_nan_or_inf=2, decr_ratio=0.5)
        self.opt = tf.train.AdamOptimizer(self.META_LR)
        self.opt = NPULossScaleOptimizer(self.opt, loss_scale_manager)
        self.opt=self.opt.minimize(self.weighted_total_losses2)
        ##end

        # self.gvs = self.opt.compute_gradients(self.weighted_total_losses2)
        # self.metatrain_op= self.opt.apply_gradients(self.gvs)

        '''Summary'''
        # self.summary_op = tf.summary.merge([tf.summary.scalar('Train Pre_update loss', self.total_loss1)]+
        #                                    [tf.summary.scalar('Train Post_update loss, step %d' % (j+1), self.total_losses2[j]) for j in range(self.TASK_ITER)]+
        #                                    [tf.summary.image('1.Input_query', tf.clip_by_value(self.inputb[0], 0., 1.),
        #                                                      max_outputs=4),
        #                                     tf.summary.image('2.output_query', tf.clip_by_value(self.outputbs[self.TASK_ITER-1][0], 0., 1.),
        #                                                      max_outputs=4),
        #                                     tf.summary.image('3.GT', self.labelb[0], max_outputs=4)
        #                                     ])


        self.saver=tf.train.Saver(max_to_keep=100000)
        pretrain_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MODEL')
        self.loader = tf.train.Saver(var_list=pretrain_vars)
        self.init=tf.global_variables_initializer()

        count_param(scope='MODEL')

        with tf.Session(config=self.conf) as sess:
            sess.run(self.init)
            if IS_TRANSFER:
                self.loader.restore(sess, TRANS_MODEL)
                print('==================== PRETRAINED MODEL Loading Succeeded ====================')

            could_load, model_step = load(self.saver, sess, self.checkpoint_dir, folder='Model%d' % self.trial)
            if could_load:
                print('Iteration:', self.step)
                print('==================== Loading Succeeded ===================================')
                assert self.step == model_step, 'The latest step and the input step do not match.'
            else:
                print('==================== No model to load ======================================')

            writer = tf.summary.FileWriter('./logs%d' % self.trial, sess.graph)

            print('[*] Training Starts')

            step = self.step
            t2 = time.time()
            while True:
                try:
                    inputa, labela, inputb, labelb = self.data_generator.make_data_tensor(sess, self.scale_list, noise_std=0.0)


                    '''feed & fetch'''
                    feed_dict = {self.inputa: inputa, self.inputb: inputb, self.labela: labela, self.labelb: labelb}

                    if step == SECOND_ORDER_GRAD_ITER:
                        second_grad=sess.run(self.second_grad_on)
                        print('1st Order Gradients: ', second_grad)

                    sess.run(self.opt, feed_dict=feed_dict)
                    step += 1
                    # print('step: ', step)
                    if step % PRINT_ITER == 0:
                        t1 = t2
                        t2 = time.time()

                        lossa_, lossb_= sess.run([self.total_loss1, self.total_losses2[-1]], feed_dict=feed_dict)
                        #
                        print('Iteration:', step, '(Pre, Post) Loss:', lossa_, lossb_, 'Time: %.2f' % (t2 - t1))
                        # 
                        # writer.add_summary(summary, step)
                        # writer.flush()

                    if step % SAVE_ITER == 0:
                        print_time()
                        save(self.saver, sess, self.checkpoint_dir, self.trial, step)

                    if step == self.META_ITER:
                        print('Done Training')
                        print_time()
                        save(self.saver, sess, self.checkpoint_dir, self.trial, step)
                        break


                except KeyboardInterrupt:
                    print('***********KEY BOARD INTERRUPT *************')
                    print('Iteration:', step)
                    print_time()
                    save(self.saver, sess, self.checkpoint_dir, self.trial, step)
                    break

    def get_loss_weights(self):
        loss_weights = tf.ones(shape=[self.TASK_ITER]) * (1.0/self.TASK_ITER)
        decay_rate = 1.0 / self.TASK_ITER / (10000 / 3)
        min_value= 0.03 / self.TASK_ITER

        loss_weights_pre = tf.maximum(loss_weights[:-1] - (tf.multiply(tf.to_float(self.global_step), decay_rate)), min_value)

        loss_weight_cur= tf.minimum(loss_weights[-1] + (tf.multiply(tf.to_float(self.global_step),(self.TASK_ITER- 1) * decay_rate)), 1.0 - ((self.TASK_ITER - 1) * min_value))
        loss_weights = tf.concat([[loss_weights_pre], [[loss_weight_cur]]], axis=1)
        return loss_weights