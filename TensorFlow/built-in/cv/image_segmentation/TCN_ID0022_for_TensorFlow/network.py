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
import numpy as np
import tensorflow as tf
from utils.data_reader import H5DataLoader, H53DDataLoader
from utils.img_utils import imsave
from utils import ops
import time

"""
This module builds a standard U-NET for semantic segmentation.
If want VAE using pixelDCL, please visit this code:
https://github.com/HongyangGao/UVAE
"""


class PixelDCN(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.def_params()
        if not os.path.exists(conf.modeldir):
            os.makedirs(conf.modeldir)
        if not os.path.exists(conf.logdir):
            os.makedirs(conf.logdir)
        if not os.path.exists(conf.sampledir):
            os.makedirs(conf.sampledir)
        self.configure_networks()
        self.train_summary = self.config_summary('train')
        self.valid_summary = self.config_summary('valid')
        self.conf.data_path = ("%s/" %(self.conf.data_path))

    def def_params(self):
        self.data_format = 'NHWC'
        if self.conf.data_type == '3D':
            self.conv_size = (3, 3, 3)
            self.pool_size = (2, 2, 2)
            self.axis, self.channel_axis = (1, 2, 3), 4
            self.input_shape = [
                self.conf.batch, self.conf.depth, self.conf.height,
                self.conf.width, self.conf.channel]
            self.output_shape = [
                self.conf.batch, self.conf.depth, self.conf.height,
                self.conf.width]
        else:
            self.conv_size = (3, 3)
            self.pool_size = (2, 2)
            self.axis, self.channel_axis = (1, 2), 3
            self.input_shape = [
                self.conf.batch, self.conf.height, self.conf.width,
                self.conf.channel]
            self.output_shape = [
                self.conf.batch, self.conf.height, self.conf.width]

    def configure_networks(self):
        self.build_network()
        if self.conf.loss_scale.strip() == "True":
            opt = npu_tf_optimizer(tf.train.AdamOptimizer(self.conf.learning_rate))
            loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,decr_every_n_nan_or_inf=2, decr_ratio=0.5)
            optimizer = NPULossScaleOptimizer(opt, loss_scale_manager)
        else:
            optimizer = npu_tf_optimizer(tf.train.AdamOptimizer(self.conf.learning_rate))

        self.train_op = optimizer.minimize(self.loss_op, name='train_op')
        tf.set_random_seed(self.conf.random_seed)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)

    def build_network(self):
        self.inputs = tf.placeholder(
            tf.float32, self.input_shape, name='inputs')
        self.labels = tf.placeholder(
            tf.int64, self.output_shape, name='labels')
        self.predictions = self.inference(self.inputs)
        self.cal_loss()

    def cal_loss(self):
        one_hot_labels = tf.one_hot(
            self.labels, depth=self.conf.class_num,
            axis=self.channel_axis, name='labels/one_hot')
        losses = tf.losses.softmax_cross_entropy(
            one_hot_labels, self.predictions, scope='loss/losses')
        self.loss_op = tf.reduce_mean(losses, name='loss/loss_op')
        self.decoded_preds = tf.argmax(
            self.predictions, self.channel_axis, name='accuracy/decode_pred')
        correct_prediction = tf.equal(
            self.labels, self.decoded_preds,
            name='accuracy/correct_pred')
        self.accuracy_op = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
            name='accuracy/accuracy_op')
        # weights = tf.cast(
        #     tf.greater(self.decoded_preds, 0, name='m_iou/greater'),
        #     tf.int32, name='m_iou/weights')
        weights = tf.cast(
            tf.less(self.labels, self.conf.channel, name='m_iou/greater'),
            tf.int64, name='m_iou/weights')
        labels = tf.multiply(self.labels, weights, name='m_iou/mul')
        self.m_iou, self.miou_op = tf.metrics.mean_iou(
            self.labels, self.decoded_preds, self.conf.class_num,
            weights, name='m_iou/m_ious')

    def config_summary(self, name):
        summarys = []
        summarys.append(tf.summary.scalar(name+'/loss', self.loss_op))
        summarys.append(tf.summary.scalar(name+'/accuracy', self.accuracy_op))
        if name == 'valid' and self.conf.data_type == '2D':
            summarys.append(
                tf.summary.image(name+'/input', self.inputs, max_outputs=100))
            summarys.append(
                tf.summary.image(
                    name+'/annotation',
                    tf.cast(tf.expand_dims(self.labels, -1),
                            tf.float32), max_outputs=100))
            summarys.append(
                tf.summary.image(
                    name+'/prediction',
                    tf.cast(tf.expand_dims(self.decoded_preds, -1),
                            tf.float32), max_outputs=100))
        summary = tf.summary.merge(summarys)
        return summary

    def inference(self, inputs):
        outputs = inputs
        down_outputs = []
        for layer_index in range(self.conf.network_depth-1):
            is_first = True if not layer_index else False
            name = 'down%s' % layer_index
            outputs = self.build_down_block(
                outputs, name, down_outputs, is_first)
        outputs = self.build_bottom_block(outputs, 'bottom')
        for layer_index in range(self.conf.network_depth-2, -1, -1):
            is_final = True if layer_index == 0 else False
            name = 'up%s' % layer_index
            down_inputs = down_outputs[layer_index]
            outputs = self.build_up_block(
                outputs, down_inputs, name, is_final)
        return outputs

    def build_down_block(self, inputs, name, down_outputs, first=False):
        out_num = self.conf.start_channel_num if first else 2 * \
            inputs.shape[self.channel_axis].value
        conv1 = ops.conv(inputs, out_num, self.conv_size,
                         name+'/conv1', self.conf.data_type)
        conv2 = ops.conv(conv1, out_num, self.conv_size,
                         name+'/conv2', self.conf.data_type)
        down_outputs.append(conv2)
        pool = ops.pool(conv2, self.pool_size, name +
                        '/pool', self.conf.data_type)
        return pool

    def build_bottom_block(self, inputs, name):
        out_num = inputs.shape[self.channel_axis].value
        conv1 = ops.conv(
            inputs, 2*out_num, self.conv_size, name+'/conv1',
            self.conf.data_type)
        conv2 = ops.conv(
            conv1, out_num, self.conv_size, name+'/conv2', self.conf.data_type)
        return conv2

    def build_up_block(self, inputs, down_inputs, name, final=False):
        out_num = inputs.shape[self.channel_axis].value
        conv1 = self.deconv_func()(
            inputs, out_num, self.conv_size, name+'/conv1',
            self.conf.data_type, action=self.conf.action)
        conv1 = tf.concat(
            [conv1, down_inputs], self.channel_axis, name=name+'/concat')
        conv2 = self.conv_func()(
            conv1, out_num, self.conv_size, name+'/conv2', self.conf.data_type)
        out_num = self.conf.class_num if final else out_num/2
        conv3 = ops.conv(
            conv2, out_num, self.conv_size, name+'/conv3', self.conf.data_type,
            not final)
        return conv3

    def deconv_func(self):
        return getattr(ops, self.conf.deconv_name)

    def conv_func(self):
        return getattr(ops, self.conf.conv_name)

    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
        if self.conf.data_type == '2D':
            train_reader = H5DataLoader(
                self.conf.data_path+self.conf.train_data)
            valid_reader = H5DataLoader(
                self.conf.data_path+self.conf.valid_data)
        else:
            train_reader = H53DDataLoader(
                self.conf.data_path+self.conf.train_data, self.input_shape)
            valid_reader = H53DDataLoader(
                self.conf.data_path+self.conf.valid_data, self.input_shape)
        for epoch_num in range(self.conf.max_step+1):
            start_time = time.time()
            if epoch_num and epoch_num % self.conf.test_interval == 0:
                inputs, labels = valid_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.labels: labels}
                loss, summary = self.sess.run(
                    [self.loss_op, self.valid_summary], feed_dict=feed_dict)
                self.save_summary(summary, epoch_num+self.conf.reload_step)
                print('----testing loss', loss)
            if epoch_num and epoch_num % self.conf.summary_interval == 0:
                inputs, labels = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.labels: labels}
                loss, _, summary = self.sess.run(
                    [self.loss_op, self.train_op, self.train_summary],
                    feed_dict=feed_dict)
                self.save_summary(summary, epoch_num+self.conf.reload_step)
            else:
                inputs, labels = train_reader.next_batch(self.conf.batch)
                feed_dict = {self.inputs: inputs,
                             self.labels: labels}
                loss, _ = self.sess.run(
                    [self.loss_op, self.train_op], feed_dict=feed_dict)
                cost_time = time.time() - start_time
                print('----training loss', loss)
                print('---sec/step',cost_time)
            if epoch_num and epoch_num % self.conf.save_interval == 0:
                self.save(epoch_num+self.conf.reload_step)

    def test(self):
        print('---->testing ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return
        if self.conf.data_type == '2D':
            test_reader = H5DataLoader(
                self.conf.data_path+self.conf.test_data, False)
        else:
            test_reader = H53DDataLoader(
                self.conf.data_path+self.conf.test_data, self.input_shape)
        self.sess.run(tf.local_variables_initializer())
        count = 0
        losses = []
        accuracies = []
        m_ious = []
        while True:
            inputs, labels = test_reader.next_batch(self.conf.batch)
            if inputs.shape[0] < self.conf.batch:
                break
            feed_dict = {self.inputs: inputs, self.labels: labels}
            loss, accuracy, m_iou, _ = self.sess.run(
                [self.loss_op, self.accuracy_op, self.m_iou, self.miou_op],
                feed_dict=feed_dict)
            print('values----->', loss, accuracy, m_iou)
            count += 1
            losses.append(loss)
            accuracies.append(accuracy)
            m_ious.append(m_iou)
        print('Loss: ', np.mean(losses))
        print('Accuracy: ', np.mean(accuracies))
        print('M_iou: ', m_ious[-1])

    def predict(self):
        print('---->predicting ', self.conf.test_step)
        if self.conf.test_step > 0:
            self.reload(self.conf.test_step)
        else:
            print("please set a reasonable test_step")
            return
        if self.conf.data_type == '2D':
            test_reader = H5DataLoader(
                self.conf.data_path+self.conf.test_data, False)
        else:
            test_reader = H53DDataLoader(
                self.conf.data_path+self.conf.test_data, self.input_shape)
        predictions = []
        while True:
            inputs, labels = test_reader.next_batch(self.conf.batch)
            if inputs.shape[0] < self.conf.batch:
                break
            feed_dict = {self.inputs: inputs, self.labels: labels}
            predictions.append(self.sess.run(
                self.decoded_preds, feed_dict=feed_dict))
        print('----->saving predictions')
        for index, prediction in enumerate(predictions):
            for i in range(prediction.shape[0]):
                imsave(prediction[i], self.conf.sampledir +
                       str(index*prediction.shape[0]+i)+'.png')

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path+'-'+str(step)
        if not os.path.exists(model_path+'.meta'):
            print('------- no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)
