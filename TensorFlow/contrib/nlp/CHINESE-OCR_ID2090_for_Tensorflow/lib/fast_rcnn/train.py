# encoding:utf-8
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
from __future__ import print_function
from npu_bridge.npu_init import *


import os
import sys
import numpy as np
import tensorflow as tf

from ..fast_rcnn.config import cfg
from ..roi_data_layer import roidb as rdl_roidb
from ..roi_data_layer.layer import RoIDataLayer
from..utils.timer import Timer
from ..rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from lib.datasets import imdb as imdb
from npu_bridge.estimator.npu import util
import lib.precision_tool.tf_config as npu_tf_config

_DEBUG = False


class SolverWrapper(object):
    def __init__(self,
                 sess,
                 network,
                 imdb,
                 roidb,
                 output_dir,
                 logdir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print('Computing bounding-box regression targets...')
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(
                roidb)
        print('done')

        # For checkpoint
        self.saver = tf.train.Saver(
            max_to_keep=1, write_version=tf.train.SaverDef.V2)
        self.writer = tf.summary.FileWriter(
            logdir=logdir, graph=tf.get_default_graph(), flush_secs=5)

    def snapshot(self, sess, iter):
        net = self.net
        if cfg.TRAIN.BBOX_REG and 'bbox_pred' in net.layers and cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(
                weights.assign(orig_0 * np.tile(self.bbox_stds,
                                                (weights_shape[0], 1))))
            sess.run(biases.assign(orig_1 * self.bbox_stds + self.bbox_means))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter + 1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)
        # save
        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        if cfg.TRAIN.BBOX_REG and 'bbox_pred' in net.layers:
            # restore net to original state
            sess.run(weights.assign(orig_0))
            sess.run(biases.assign(orig_1))

    def build_image_summary(self):
        # A simple graph for write image summary

        log_image_data = tf.placeholder(tf.uint8, [None, None, 3])
        log_image_name = tf.placeholder(tf.string)
        # import tensorflow.python.ops.gen_logging_ops as logging_ops
        from tensorflow.python.ops import gen_logging_ops
        from tensorflow.python.framework import ops as _ops
        log_image = gen_logging_ops.image_summary(
            log_image_name, tf.expand_dims(log_image_data, 0), max_images=1)
        _ops.add_to_collection(_ops.GraphKeys.SUMMARIES, log_image)
        # log_image = tf.summary.image(log_image_name, tf.expand_dims(log_image_data, 0), max_outputs=1)
        return log_image, log_image_data, log_image_name

    def train_model(self, sess, max_iters, restore=False):
        """Network training loop."""
        _feat_stride = [16, ]
        anchor_scales = cfg.ANCHOR_SCALES
        input = list()
        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)


        total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = self.net.build_loss(
            ohem=cfg.TRAIN.OHEM)
        # scalar summary
        tf.summary.scalar('rpn_reg_loss', rpn_loss_box)
        tf.summary.scalar('rpn_cls_loss', rpn_cross_entropy)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)
        summary_op = tf.summary.merge_all()

        # log_image, log_image_data, log_image_name = \
        #     self.build_image_summary()
        # optimizer
        lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        if cfg.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.TRAIN.LEARNING_RATE)
        elif cfg.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.TRAIN.LEARNING_RATE)
        else:
            # lr = tf.Variable(0.0, trainable=False)
            momentum = cfg.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(lr, momentum)
        global_step = tf.Variable(0, trainable=False)
        
        loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32,
                                                               incr_every_n_steps=1000,
                                                               decr_every_n_nan_or_inf=2,
                                                               decr_ratio=0.8)
        opt = NPULossScaleOptimizer(opt, loss_scale_manager, is_distributed=False) # 开启loss_scale

        with_clip = True
        if with_clip:
            # scale
            scale_vv = loss_scale_manager.get_loss_scale()
            scale_total_loss = total_loss * scale_vv
            tvars = tf.trainable_variables()
            grads = tf.gradients(scale_total_loss, tvars)
            gg = []
            for g in grads:
                gg.append(g / scale_vv)
            grads = gg
            grads, norm = tf.clip_by_global_norm(
                grads, 10.0)
            train_op = opt.apply_gradients(
                list(zip(grads, tvars)), global_step=global_step)
        else:
            train_op = opt.minimize(total_loss, global_step=global_step)

        # intialize variables
        sess.run(tf.global_variables_initializer())
        restore_iter = 0

        # load vgg16
        if self.pretrained_model is not None and not restore:
            try:
                print(('Loading pretrained model '
                       'weights from {:s}').format(self.pretrained_model))
                self.net.load(self.pretrained_model, sess, True)
            except:
                raise 'Check your pretrained model {:s}'.format(
                    self.pretrained_model)

        # resuming a trainer
        if restore:
            # try:
            print('output_dir:', self.output_dir)
            # 加载ckpt文件路径，而非指向checkpoint
            ckpt = tf.train.get_checkpoint_state(
                self.output_dir + '/')
            print(
                'Restoring from {}...'.format(ckpt.model_checkpoint_path),
                end=' ')
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            stem = os.path.splitext(
                os.path.basename(ckpt.model_checkpoint_path))[0]
            restore_iter = int(stem.split('_')[-1])
            sess.run(global_step.assign(restore_iter))
            print('done')
            # except:

            # raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

        last_snapshot_iter = -1
        timer = Timer()
        print(restore_iter, max_iters)
        for iter in range(restore_iter, max_iters):
            timer.tic()
            # learning rate
            print(iter)
            if iter != 0 and iter % cfg.TRAIN.STEPSIZE == 0:
                sess.run(tf.assign(lr, lr.eval() * cfg.TRAIN.GAMMA))
                print(lr)

            # get one batch
            blobs = data_layer.forward()
            feed_dict1 = {
                self.net.data: blobs['data']
            }

            # sess.run(tf.global_variables_initializer())

            fetch_list = ["rpn_cls_score/Reshape_1:0"]

            rpn_cls_score= sess.run("rpn_cls_score/Reshape_1:0", feed_dict=feed_dict1)

            input.clear()
            input.append(rpn_cls_score)
            input.append(blobs['gt_boxes'])
            input.append(blobs['gt_ishard'])
            input.append(blobs['dontcare_areas'])
            input.append(blobs['im_info'])

            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
                anchor_target_layer_py(rpn_cls_score,blobs['gt_boxes'],blobs['gt_ishard'],\
                    blobs['dontcare_areas'],blobs['im_info'], _feat_stride, anchor_scales)

            # sess.run(tf.global_variables_initializer())
            feed_dict = {
                self.net.data: blobs['data'],
                self.net.keep_prob: 0.5,
                self.net.rpn_labels: rpn_labels,
                self.net.rpn_bbox_targets: rpn_bbox_targets,
                self.net.rpn_bbox_inside_weights: rpn_bbox_inside_weights,
                self.net.rpn_bbox_outside_weights: rpn_bbox_outside_weights
            }

            res_fetches = []
            fetch_list = [
                total_loss, model_loss, rpn_cross_entropy, rpn_loss_box,
                summary_op, train_op
            ] + res_fetches

            total_loss_val, model_loss_val, rpn_loss_cls_val, rpn_loss_box_val, \
            summary_str, _ = sess.run(fetches=fetch_list, feed_dict=feed_dict)
            self.writer.add_summary(
                summary=summary_str, global_step=global_step.eval())

            _diff_time = timer.toc(average=False)

            if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                # print(
                #     'iter: %d / %d, model loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, lr: %f' % \
                #     (iter, max_iters, model_loss_val, rpn_loss_cls_val, rpn_loss_box_val, lr.eval()))
                print('iter: %d / %d'% (iter, max_iters))
                print('total loss: %.4f'% total_loss_val)
                print('model loss: %.4f'% model_loss_val)
                print('cls_rpn_loss: %.4f'% rpn_loss_cls_val)
                print('box_rpn_loss: %.4f'% rpn_loss_box_val)
                print('speed: {:.3f}'.format(_diff_time))

            if (iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    if cfg.TRAIN.HAS_RPN:
        rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            # obsolete
            # layer = GtDataLayer(roidb)
            raise "Calling caffe modules..."
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer


def train_net(network,
              imdb,
              roidb,
              output_dir,
              log_dir,
              pretrained_model=None,
              max_iters=40000,
              restore=False):
    """Train a Fast R-CNN network."""

    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF # 必须显式关闭
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF # 必须显式关闭
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp32") # force_fp32
    # # dump_path：dump数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限
    # custom_op.parameter_map["dump_path"].s = tf.compat.as_bytes("/home/data/test_user04/ID2090/dump_path") 
    # # enable_dump_debug：是否开启溢出检测功能
    # custom_op.parameter_map["enable_dump_debug"].b = False
    # # dump_debug_mode：溢出检测模式，取值：all/aicore_overflow/atomic_overflow
    # custom_op.parameter_map["dump_debug_mode"].s = tf.compat.as_bytes("all") 
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.75
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(
            sess,
            network,
            imdb,
            roidb,
            output_dir,
            logdir=log_dir,
            pretrained_model=pretrained_model)
        print('Solving...')
        sw.train_model(sess, max_iters, restore=restore)
        print('done solving')
