#! /usr/bin/env python
# -*- coding: utf-8 -*-
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
# ==============================================================================
from npu_bridge.npu_init import *
from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer

import sys
import os
import time
import shutil
import numpy as np
from absl import flags
import cv2

import threading
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import OrderedDict
import itertools

import core.utils as utils
from core.dataset import Dataset, DatasetBatchFetcher
from core.yolov3_tiny import YOLOV3Tiny
from core.yolov3 import YOLOV3
from core.yolov4 import YOLOV4
from core.yolov5 import YOLOV5
from core.config import cfg

import tensorflow
print('tensorflow.version=', tensorflow.__version__)
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

###############################################################################

flags.DEFINE_string('npu_precision_mode', cfg.NPU.PRECISION_MODE, 'NPU precision mode.')
flags.DEFINE_integer('npu_loss_scale_flag', cfg.NPU.LOSS_SCALE_FLAG, 'NPU loss-scale flag.')
flags.DEFINE_float ('npu_loss_scale_value', cfg.NPU.LOSS_SCALE, 'NPU loss-scale value.')
flags.DEFINE_bool  ('npu_overflow_dump', cfg.NPU.OVERFLOW_DUMP, 'NPU overflow dump.')

flags.DEFINE_string('net_type', 'yolov5', 'yolov3/4/5')
flags.DEFINE_string('exec_mode', 'train', 'train | eval')
flags.DEFINE_bool  ('eval_after_training', True, 'True | False')

flags.DEFINE_string('data_classes_file', cfg.YOLO.CLASSES, 'File path of label classes.')
flags.DEFINE_string('data_annotations_file', cfg.TRAIN.ANNOT_PATH, 'File path of annotations.')

flags.DEFINE_integer('train_worker_num', 5, 'Worker number for training.')
flags.DEFINE_integer('eval_worker_num', 5, 'Worker number for evaluating.')
flags.DEFINE_integer('train_batch_size', cfg.TRAIN.BATCH_SIZE, 'Batch size for training.')
flags.DEFINE_float  ('learning_rate_init', cfg.TRAIN.LEARN_RATE_INIT, 'Initial learning rate.')
flags.DEFINE_float  ('learning_rate_end', cfg.TRAIN.LEARN_RATE_END, 'Final learning rate.')

flags.DEFINE_integer('warmup_epochs', cfg.TRAIN.WARMUP_EPOCHS, 'Epoch number of Warmup.')
flags.DEFINE_integer('first_stage_epochs', cfg.TRAIN.FIRST_STAGE_EPOCHS, 'Epoch number of first-stage training.')
flags.DEFINE_integer('second_stage_epochs', cfg.TRAIN.SECOND_STAGE_EPOCHS, 'Epoch number of second-stage training.')
flags.DEFINE_float  ('max_total_steps', cfg.TRAIN.MAX_TOTAL_STEPS, 'Max training step number.')

flags.DEFINE_string('initial_ckpt_path', cfg.TRAIN.INITIAL_WEIGHT, 'Initial ckpt path.')
flags.DEFINE_integer('begin_epoch', 0, 'Beginning epoch.')

flags.DEFINE_float('eval_max_steps', cfg.TEST.MAX_STEPS, 'Max eval step number.')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

print("\n================ FLAGS:")
for item in FLAGS:
    print(item, ':', FLAGS[item].value)

cfg.NPU.PRECISION_MODE  = FLAGS.npu_precision_mode
cfg.NPU.LOSS_SCALE_FLAG = FLAGS.npu_loss_scale_flag
cfg.NPU.LOSS_SCALE      = FLAGS.npu_loss_scale_value
cfg.NPU.OVERFLOW_DUMP   = FLAGS.npu_overflow_dump

cfg.YOLO.CLASSES        = FLAGS.data_classes_file
cfg.TRAIN.ANNOT_PATH    = FLAGS.data_annotations_file

cfg.TRAIN.BATCH_SIZE      = FLAGS.train_batch_size
cfg.TRAIN.LEARN_RATE_INIT = FLAGS.learning_rate_init
cfg.TRAIN.LEARN_RATE_END  = FLAGS.learning_rate_end

cfg.TRAIN.WARMUP_EPOCHS       = FLAGS.warmup_epochs
cfg.TRAIN.FIRST_STAGE_EPOCHS  = FLAGS.first_stage_epochs
cfg.TRAIN.SECOND_STAGE_EPOCHS = FLAGS.second_stage_epochs
cfg.TRAIN.MAX_TOTAL_STEPS     = FLAGS.max_total_steps

cfg.TRAIN.INITIAL_WEIGHT      = FLAGS.initial_ckpt_path

cfg.TEST.MAX_STEPS = FLAGS.eval_max_steps

import json
beautiful_cfg = json.dumps(cfg, indent=2, ensure_ascii=False)
print("\n================ CONFIGURATION:")
print(beautiful_cfg)

###############################################################################

class MixedPrecisionOptimizer(tf.train.Optimizer):
    """An optimizer that updates trainable variables in fp32."""

    def __init__(self, optimizer,
                 scale=None,
                 name="MixedPrecisionOptimizer",
                 use_locking=False):
        super(MixedPrecisionOptimizer, self).__init__(
            name=name, use_locking=use_locking)
        self._optimizer = optimizer
        self._scale = float(scale) if scale is not None else 1.0

    def compute_gradients(self, loss, var_list=None, *args, **kwargs):
        if var_list is None:
            var_list = (
                    tf.trainable_variables() +
                    tf.get_collection(tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))

        replaced_list = var_list

        if self._scale != 1.0:
            loss = tf.scalar_mul(self._scale, loss)

        gradvar = self._optimizer.compute_gradients(loss, replaced_list, *args, **kwargs)

        final_gradvar = []
        for orig_var, (grad, var) in zip(var_list, gradvar):
            if var is not orig_var:
                grad = tf.cast(grad, orig_var.dtype)
            if self._scale != 1.0:
                grad = tf.scalar_mul(1. / self._scale, grad)
            final_gradvar.append((grad, orig_var))

        return final_gradvar

    def apply_gradients(self, *args, **kwargs):
        return self._optimizer.apply_gradients(*args, **kwargs)

###############################################################################

def broadcast_variables(vars, root_rank):
    """Broadcasts variables from root rank to all other processes.
    Arguments:
        root_rank: rank of the process from which global variables will be broadcasted
        to all other processes.
    """
    op_list = []
    for var in vars:
        inputs = [var]
        outputs=hccl_ops.broadcast(tensor=inputs, root_rank=root_rank)
        if outputs is not None:
            op_list.append(outputs[0].op)
            op_list.append(tf.assign(var, outputs[0]))

    return tf.group(op_list)


def broadcast_global_variables(root_rank):
    """Broadcasts all global variables from root rank to all other processes.
    Arguments:
        root_rank: rank of the process from which global variables will be broadcasted
        to all other processes.
    """
    return broadcast_variables(vars=tf.trainable_variables(), root_rank=root_rank)
    
###############################################################################

class YoloTrain(object):
    def __init__(self, net_type):
        self.net_type = net_type
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs = cfg.TRAIN.FIRST_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS
        self.initial_weight = cfg.TRAIN.INITIAL_WEIGHT
        
        self.npu_precision_mode = cfg.NPU.PRECISION_MODE
        self.npu_loss_scale = cfg.NPU.LOSS_SCALE

        self.ckpt_path = cfg.TRAIN.CKPT_PATH        
        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
        
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale = 150

        self.log_path = ('log/%s' % net_type)
        if os.path.exists(self.log_path):
            shutil.rmtree(self.log_path)
            #os.removedirs(self.log_path)
        os.makedirs(self.log_path)

        shard_num = int(os.environ["RANK_SIZE"])
        shard_id = int(os.environ["RANK_ID"]) - int(os.environ["RANK_ID_START"])

        self.trainset = Dataset('train', self.net_type, shard_num=shard_num, shard_id=shard_id)
        self.testset = Dataset('test', self.net_type, shard_num=shard_num, shard_id=shard_id)
        self.steps_per_period = len(self.trainset)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes(self.npu_precision_mode)

        self.sess = tf.Session(config=npu_config_proto(config_proto=config))

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope(self.net_type):
            if self.net_type == 'tiny':
                self.model = YOLOV3Tiny(self.input_data, self.trainable)
            elif self.net_type == 'yolov3':
                self.model = YOLOV3(self.input_data, self.trainable)
            elif self.net_type == 'yolov4':
                self.model = YOLOV4(self.input_data, self.trainable)
            elif self.net_type == 'yolov5':
                self.model = YOLOV5(self.input_data, self.trainable)
            else:
                print('self.net_type=%s error' % self.net_type)

    def __init_train(self):
        with tf.name_scope('input'):
            if net_type == 'tiny':
                self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
                self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')

                self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
                self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')

            else:                
                self.label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
                self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
                self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')

                self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
                self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
                self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')

        with tf.name_scope('define_loss'):
            if self.net_type == 'tiny':
                self.net_var = tf.global_variables()
                self.iou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(self.label_mbbox, self.label_lbbox,
                                                                                        self.true_mbboxes, self.true_lbboxes)
                self.loss = self.iou_loss + self.conf_loss + self.prob_loss

            elif self.net_type == 'yolov3':
                self.net_var = tf.global_variables()
                self.iou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(self.label_sbbox, self.label_mbbox, self.label_lbbox,
                                                                                        self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
                self.loss = self.iou_loss + self.conf_loss + self.prob_loss
            
            elif self.net_type == 'yolov4' or self.net_type == 'yolov5':
                iou_use = 2  # (0, 1, 2) ==> (giou_loss, diou_loss, ciou_loss)
                focal_use = True  # (False, True) ==> (normal, focal_loss)
                label_smoothing = 0

                self.net_var = tf.global_variables()
                self.iou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(self.label_sbbox, self.label_mbbox, self.label_lbbox,
                                                                                        self.true_sbboxes, self.true_mbboxes, self.true_lbboxes,
                                                                                        iou_use, focal_use, label_smoothing)
                #self.loss = self.iou_loss + self.conf_loss + self.prob_loss
                self.loss = self.iou_loss * 0.05 + self.conf_loss * 1.0 + self.prob_loss * 0.5
                
            else:
                print('self.net_type=%s error' % self.net_type)

        with tf.name_scope('l2_regularizer'):
            l2_reg_loss = 0
            for var in tf.global_variables():
                if 'weight' in var.name:
                    print("L2 on:", var.name)
                    l2_reg_loss += tf.reduce_sum(var**2) / 2
            self.loss += l2_reg_loss * cfg.TRAIN.WEIGHT_L2_REG_COEF
        
        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period, dtype=tf.float32, name='warmup_steps')
            train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period,
                                       dtype=tf.float32, name='train_steps')
            
            self.learn_rate = tf.cond(pred=self.global_step < warmup_steps, true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                                      false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) * \
                                              (1 + tf.cos((self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi)))
            global_step_update = tf.assign_add(self.global_step, 1.0)

        # with tf.name_scope('define_weight_decay'):
        #     moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope('define_first_stage_train'):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if net_type == 'tiny':
                    bboxes = ['conv_mbbox', 'conv_lbbox']
                else:
                    bboxes = ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']
                
                if var_name_mess[0] in bboxes:
                    self.first_stage_trainable_var_list.append(var)

            optimizer = tf.train.AdamOptimizer(self.learn_rate,
                    beta1=cfg.TRAIN.ADAM_BETA1, beta2=cfg.TRAIN.ADAM_BETA2, epsilon=cfg.TRAIN.ADAM_EPSILON)
            #optimizer = MixedPrecisionOptimizer(optimizer, scale=self.npu_loss_scale)
            if int(os.environ["RANK_SIZE"]) > 1:
                optimizer = NPUDistributedOptimizer(optimizer)

            first_stage_optimizer = optimizer.minimize(self.loss, var_list=self.first_stage_trainable_var_list)
            
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    # with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope('define_second_stage_train'):
            second_stage_trainable_var_list = tf.trainable_variables()

            optimizer = tf.train.AdamOptimizer(self.learn_rate,
                    beta1=cfg.TRAIN.ADAM_BETA1, beta2=cfg.TRAIN.ADAM_BETA2, epsilon=cfg.TRAIN.ADAM_EPSILON)
            #optimizer = MixedPrecisionOptimizer(optimizer, scale=self.npu_loss_scale)
            if int(os.environ["RANK_SIZE"]) > 1:
                optimizer = NPUDistributedOptimizer(optimizer)

            second_stage_optimizer = optimizer.minimize(self.loss, var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    # with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

        with tf.name_scope('summary'):
            tf.summary.scalar('learn_rate', self.learn_rate)
            tf.summary.scalar('iou_loss', self.iou_loss)
            tf.summary.scalar('conf_loss', self.conf_loss)
            tf.summary.scalar('prob_loss', self.prob_loss)
            tf.summary.scalar('total_loss', self.loss)

            self.write_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.log_path, graph=self.sess.graph)


    def train(self):
        self.__init_train()

        self.sess.run(tf.global_variables_initializer())

        if self.initial_weight :
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)

            begin_epoch = FLAGS.begin_epoch
            self.sess.run(tf.assign(self.global_step, 1 + begin_epoch*self.steps_per_period))
        else:
            self.first_stage_epochs = 0
            begin_epoch = 0

        print('=> Begin training from epoch %d' % begin_epoch)

        self.rank_id = int(os.environ["RANK_ID"])
        self.root_rank = int(os.environ["RANK_ID_START"])
        if int(os.environ["RANK_SIZE"]) > 1:
            bcast_op = broadcast_global_variables(root_rank=self.root_rank)
            self.sess.run(bcast_op)

        mutex_work = threading.Lock()
        mutex_data_iter = threading.Lock()
        mutex_sess_run = threading.Lock()

        #saving = 0.0
        self.pre_step_time = time.time()
        self.global_step_val = -1
        self.avg_steps_per_sec = 0.

        for epoch in range(begin_epoch, (1 + self.first_stage_epochs + self.second_stage_epochs)):
            if epoch < self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            train_epoch_loss = []

            self.trainset.rewind()
            iter_data = enumerate(self.trainset)
            epoch_step_num = len(self.trainset)
            
            def worker_step(tid, batch_fetcher):

                with mutex_data_iter:
                    try:
                        epoch_step, batch_fetcher.batch_annotations = next(iter_data)
                    except StopIteration:
                        return False
                    if batch_fetcher.batch_annotations is None:
                        return False
                
                train_data = batch_fetcher.process()

                with mutex_sess_run:

                    if net_type == 'tiny':
                        _, summary, train_step_loss, global_step_val = self.sess.run(
                            [train_op, self.write_op, self.loss, self.global_step], 
                            feed_dict={self.input_data: train_data[0],
                                    self.label_mbbox: train_data[1], self.label_lbbox: train_data[2],
                                    self.true_mbboxes: train_data[3], self.true_lbboxes: train_data[4], 
                                    self.trainable: [True],})
                    else:
                        _, summary, train_step_loss, global_step_val = self.sess.run(
                            [train_op, self.write_op, self.loss, self.global_step], 
                            feed_dict={self.input_data: train_data[0],
                                    self.label_sbbox: train_data[1], self.label_mbbox: train_data[2], self.label_lbbox: train_data[3],
                                    self.true_sbboxes: train_data[4], self.true_mbboxes: train_data[5], self.true_lbboxes: train_data[6], 
                                    self.trainable: [True],}) 

                    sess_run_end = time.time()
                    step_interval = sess_run_end - self.pre_step_time
                    self.pre_step_time = sess_run_end
                    steps_per_sec = 1/step_interval
                    self.avg_steps_per_sec = (steps_per_sec if self.avg_steps_per_sec==0
                                              else self.avg_steps_per_sec * 0.99 + steps_per_sec * 0.01)
                    
                    global_step_val = self.global_step.eval(self.sess)
                    global_step_val = np.round(global_step_val)

                    print('Device %d - Epoch %d - Step %d - train loss: %.2f - %d/%d - %.3fs - %.3fit/s' 
                            %(self.rank_id, epoch, global_step_val, train_step_loss, epoch_step, 
                              epoch_step_num, step_interval, self.avg_steps_per_sec), end='\r')

                    train_epoch_loss.append(train_step_loss)
                    self.summary_writer.add_summary(summary, global_step_val)

                    self.global_step_val = global_step_val

                if cfg.TRAIN.MAX_TOTAL_STEPS is not None:
                    if global_step_val >= cfg.TRAIN.MAX_TOTAL_STEPS:
                        return False

                return True

            def worker_epoch(tid):
                with mutex_work:
                    if not hasattr(self, 'batch_fetchers'):
                        self.batch_fetchers = dict()
                    if not tid in self.batch_fetchers.keys():
                        self.batch_fetchers[tid] = DatasetBatchFetcher(self.trainset)
                    batch_fetcher = self.batch_fetchers[tid]

                while worker_step(tid, batch_fetcher):
                    pass

            threads = []
            for i in range(FLAGS.train_worker_num):
                t = threading.Thread(target=worker_epoch, args=(i,))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

            train_epoch_loss = np.mean(train_epoch_loss)
            
            print("")
            ckpt_file = os.path.join(self.ckpt_path, '%s_epoch_%d_trainloss_%.4f.ckpt' % (
                    self.net_type, epoch, train_epoch_loss))
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            if True: # saving <= 0.0 or saving > train_epoch_loss:
                print('=> Epoch: %2d Time: %s Train loss: %.2f Saving %s ...' % 
                     (epoch, log_time, train_epoch_loss, ckpt_file))
                self.saver.save(self.sess, ckpt_file, global_step=self.global_step)
                saving = train_epoch_loss
            
            else:
                print('=> Epoch: %2d Time: %s Train loss: %.2f NO Saving' % (epoch, log_time, train_epoch_loss))

            if cfg.TRAIN.MAX_TOTAL_STEPS is not None:
                if self.global_step_val >= cfg.TRAIN.MAX_TOTAL_STEPS:
                    break
            
            print("")

        if FLAGS.eval_after_training :
            self.__eval()

    def __predict(self, score_threshold, nms_iou_threshold, draw_bbox=False):

        mutex_work = threading.Lock()
        mutex_data_iter = threading.Lock()
        mutex_sess_run = threading.Lock()
        mutex_json = threading.Lock()

        supercategory = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, \
            17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, \
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, \
            55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, \
            76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        json_list = []
        
        self.testset.rewind()
        iter_data = enumerate(self.testset)
        epoch_step_num = len(self.testset)

        self.pre_step_time = time.time()
        self.avg_steps_per_sec = 0.

        def worker_step(batch_fetcher):

            with mutex_data_iter:
                try:
                    epoch_step, batch_fetcher.batch_annotations = next(iter_data)
                except StopIteration:
                    return False
                if batch_fetcher.batch_annotations is None:
                    return False

            test_data = batch_fetcher.process()

            batch_image_id, batch_scale, batch_dw, batch_dh, batch_image_path = test_data[7:12]

            with mutex_sess_run:
                if net_type == 'tiny':
                    pred_mbbox, pred_lbbox = self.sess.run(
                        [self.model.pred_mbbox, self.model.pred_lbbox],
                        feed_dict={self.input_data: test_data[0], self.trainable: [False]}
                    )
                else:
                    pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
                        [self.model.pred_sbbox, self.model.pred_mbbox, self.model.pred_lbbox],
                        feed_dict={self.input_data: test_data[0], self.trainable: [False]}
                    )
                
                sess_run_end = time.time()
                step_interval = sess_run_end - self.pre_step_time
                self.pre_step_time = sess_run_end
                steps_per_sec = 1/step_interval
                self.avg_steps_per_sec = (steps_per_sec if self.avg_steps_per_sec==0
                                          else self.avg_steps_per_sec * 0.99 + steps_per_sec * 0.01)
                
                print('[Test] %d/%d - %.3fs - %.3fit/s' 
                        %(epoch_step, epoch_step_num, step_interval, self.avg_steps_per_sec), end='\r')

            with mutex_json:
                for batch in range(self.testset.batch_size):
                    image_id, scale, dw, dh = batch_image_id[batch], batch_scale[batch], batch_dw[batch], batch_dh[batch]
                    image_path = batch_image_path[batch]

                    for i, pred in enumerate((pred_sbbox[batch:batch+1], pred_mbbox[batch:batch+1], pred_lbbox[batch:batch+1])):
                        scores_ = pred[..., 4:5].flatten()
                        probs_ = np.max(pred[..., 5:85], axis=-1).flatten()
                        categories_ = np.argmax(pred[..., 5:85], axis=-1).flatten()
                        bboxes_ = pred[..., 0:4].reshape(-1,4)

                        scores = scores_ if i==0 else np.concatenate((scores, scores_), axis=0)
                        probs = probs_ if i==0 else np.concatenate((probs, probs_), axis=0)
                        categories = categories_ if i==0 else np.concatenate((categories, categories_), axis=0)
                        bboxes = bboxes_ if i==0 else np.concatenate((bboxes, bboxes_), axis=0)

                    scores = scores * probs

                    # max detections for one picture
                    mask = (scores > score_threshold)
                    scores = scores[mask]
                    categories = categories[mask]
                    bboxes = bboxes[mask]

                    # scale bboxes to origin shape
                    bboxes[..., 0] = (bboxes[..., 0] - dw) / scale # xcenter
                    bboxes[..., 1] = (bboxes[..., 1] - dh) / scale # ycenter
                    bboxes[..., [2,3]] = bboxes[..., [2,3]] / scale # w, h

                    w = bboxes[..., 2]
                    h = bboxes[..., 3]
                    xmin = bboxes[..., 0] - w / 2
                    ymin = bboxes[..., 1] - h / 2
                    xmax = bboxes[..., 0] + w / 2
                    ymax = bboxes[..., 1] + h / 2

                    bboxes_for_nms = np.stack((xmin, ymin, xmax, ymax, scores, categories), axis=1)
                    bboxes_after_nms = utils.nms(bboxes_for_nms, iou_type='iou', iou_threshold=nms_iou_threshold)
                    if len(bboxes_after_nms) == 0 :
                        continue

                    xmin, ymin, xmax, ymax, scores, categories = np.stack(bboxes_after_nms, axis=1)
                    categories = categories.astype(np.int32)
                    w, h = xmax-xmin, ymax-ymin

                    if draw_bbox:
                        image = cv2.imread(image_path)
                        image = utils.draw_bbox(image, bboxes_after_nms)
                        out_file = "predict_result_%012d.jpg"%image_id
                        cv2.imwrite(out_file, image)
                    
                    for n in range(len(scores)):
                        coco_dict = OrderedDict()
                        coco_dict["score"] = float(scores[n])
                        coco_dict["image_id"] = int(image_id)
                        cid = categories[n]
                        coco_dict["category_id"] = supercategory[cid]
                        coco_dict["bbox"] = [float(xmin[n]), float(ymin[n]), float(w[n]), float(h[n])]
                        json_list.append(coco_dict)

            return True

        self.eval_step = 0

        def worker_epoch(tid):
            with mutex_work:
                if not hasattr(self, 'batch_fetchers'):
                    self.batch_fetchers = dict()
                if not tid in self.batch_fetchers.keys():
                    self.batch_fetchers[tid] = DatasetBatchFetcher(self.testset)
                batch_fetcher = self.batch_fetchers[tid]

            while worker_step(batch_fetcher):
                self.eval_step += 1

                if cfg.TEST.MAX_STEPS is not None:
                    if self.eval_step >= cfg.TEST.MAX_STEPS:
                        break

        threads = []
        for i in range(FLAGS.eval_worker_num):
            t = threading.Thread(target=worker_epoch, args=(i,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        print("")
        
        return json_list

    def __eval(self):
        json_list = self.__predict(score_threshold=0.1, nms_iou_threshold=0.5)

        dtfile = './det_result.json'

        print('writing into json file ...')
        with open(dtfile, 'w', encoding='utf-8') as f:
            json.dump(json_list, f, ensure_ascii=False)

        eval_json(dtfile)

    def __load_predict_model(self):
        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(tf.global_variables())

        print('=> Restoring weights from: %s ... ' % self.initial_weight)
        self.loader.restore(self.sess, self.initial_weight)
    
    def predict(self):
        self.__load_predict_model()
        self.__predict(score_threshold=0.6, nms_iou_threshold=0.5, draw_bbox=True)

    def eval(self):
        self.__load_predict_model()
        self.__eval()

def eval_json(dtfile):
    with open(dtfile, 'r', encoding='utf-8') as f:
        j = json.load(f)
    imgIds = list(set([item["image_id"] for item in j]))

    cocoGT = COCO(cfg.TEST.ANNOT_PATH_)
    cocoDT = cocoGT.loadRes(dtfile)
    cocoEval = COCOeval(cocoGT, cocoDT, 'bbox')
    if imgIds is not None:
        cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # AP of every class
    precisions = cocoEval.eval["precision"]
    with open('data/classes/coco.names', 'r') as f:
        class_names = [line.strip() for line in f]
    results_per_category = []
    for idx, name in enumerate(class_names):
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        results_per_category.append(("{}".format(name), float(ap*100)))

    N_COLS = min(6, len(results_per_category) * 2)
    results_flatten = list(itertools.chain(*results_per_category))
    results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
    print(list(results_2d))


if __name__ == '__main__':

    net_type = FLAGS.net_type
    
    if FLAGS.exec_mode == "train":
        YoloTrain(net_type).train()
    elif FLAGS.exec_mode == "eval":
        YoloTrain(net_type).eval()
    elif FLAGS.exec_mode == "eval_json":
        dtfile = './det_result.json'
        eval_json(dtfile)
    elif FLAGS.exec_mode == "test_image":
        YoloTrain(net_type).predict()
    else:
        print("Unknown exec_mode :", FLAGS.exec_mode)
