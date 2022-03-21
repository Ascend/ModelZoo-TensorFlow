#!/usr/bin/env python3
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
from Train.config import get_logging_config, args, train_dir
from Train.config import config as net_config
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

import time
import os
import logging.config
import tensorflow as tf
import numpy as np

from Train.voc_loader import VOC_CATS
from Train.resnet import ResNet
from Train.utils import print_variables
from Train.utils_tf import yxyx_to_xywh, data_augmentation
from Train.boxer import PriorBoxGrid
import matplotlib
matplotlib.use('Agg')

slim = tf.contrib.slim
streaming_mean_iou = tf.contrib.metrics.streaming_mean_iou

logging.config.dictConfig(get_logging_config(args.run_name))
log = logging.getLogger()

dataset_num_classes = len(VOC_CATS)


def npu_tf_optimizer(opt):
    npu_opt = NPUDistributedOptimizer(opt)
    return npu_opt


def objective(location, confidence, refine_ph, classes_ph,
              pos_mask, seg_logits, seg_gt, config):
    def smooth_l1(x, y):
        abs_diff = tf.abs(x-y)
        tf.reduce_sum([[1, 0], [0, 1]], 1)

        return tf.reduce_sum(tf.where(abs_diff < 1, 0.5*abs_diff*abs_diff, abs_diff - 0.5),
                             1)

    def segmentation_loss(seg_logits, seg_gt, config):
        mask = seg_gt <= dataset_num_classes
        seg_logits = tf.boolean_mask(seg_logits, mask)
        seg_gt = tf.boolean_mask(seg_gt, mask)
        seg_predictions = tf.argmax(seg_logits, axis=1)
        seg_loss_local = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_logits,
                                                                        labels=seg_gt)
        seg_loss = tf.reduce_mean(seg_loss_local)
        mean_iou, update_mean_iou = tf.metrics.mean_iou(seg_gt, seg_predictions, dataset_num_classes)
        return seg_loss, mean_iou, update_mean_iou

    def detection_loss(location, confidence, refine_ph, classes_ph, pos_mask):
        neg_mask = tf.logical_not(pos_mask)
        number_of_positives = tf.reduce_sum(tf.to_int32(pos_mask))
        true_number_of_negatives = tf.minimum(3 * number_of_positives,
                                              tf.shape(pos_mask)[1] - number_of_positives)
        # max is to avoid the case where no positive boxes were sampled
        number_of_negatives = tf.maximum(1, true_number_of_negatives)
        num_pos_float = tf.to_float(tf.maximum(1, number_of_positives))
        normalizer = tf.to_float(tf.add(number_of_positives, number_of_negatives))
        # tf.summary.scalar('batch/size', normalizer)

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=confidence,
                                                                       labels=classes_ph)

        pos_class_loss = tf.reduce_sum(tf.boolean_mask(cross_entropy, pos_mask))

        top_k_worst, top_k_inds = tf.nn.top_k(tf.boolean_mask(cross_entropy, neg_mask),
                                              number_of_negatives)
        # multiplication is to avoid the case where no positive boxes were sampled
        neg_class_loss = tf.reduce_sum(top_k_worst) * \
                         tf.cast(tf.greater(true_number_of_negatives, 0), tf.float32)
        class_loss = (neg_class_loss + pos_class_loss) / num_pos_float
        # cond is to avoid the case where no positive boxes were sampled
        bbox_loss = tf.where(tf.equal(tf.reduce_sum(tf.cast(pos_mask, tf.int32)), 0), 0.0,
                             tf.reduce_mean(smooth_l1(tf.boolean_mask(location, pos_mask),
                                                             tf.boolean_mask(refine_ph, pos_mask))))

        # tf.summary.scalar('loss/bbox', bbox_loss)

        inferred_class = tf.cast(tf.argmax(confidence, 2), tf.int32)
        positive_matches = tf.equal(tf.boolean_mask(inferred_class, pos_mask),
                                    tf.boolean_mask(classes_ph, pos_mask))
        hard_matches = tf.equal(tf.boolean_mask(inferred_class, neg_mask),
                                tf.boolean_mask(classes_ph, neg_mask))
        hard_matches = tf.gather(hard_matches, top_k_inds)
        train_acc = ((tf.reduce_sum(tf.to_float(positive_matches)) +
                      tf.reduce_sum(tf.to_float(hard_matches))) / normalizer)
        # tf.summary.scalar('accuracy/train', train_acc)

        recognized_class = tf.argmax(confidence, 2)
        tp = tf.reduce_sum(tf.to_float(tf.logical_and(recognized_class > 0, pos_mask)))
        fp = tf.reduce_sum(tf.to_float(tf.logical_and(recognized_class > 0, neg_mask)))
        fn = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(recognized_class, 0), pos_mask)))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2*(precision * recall)/(precision + recall)
        return class_loss, bbox_loss, train_acc

    the_loss = 0
    train_acc = tf.constant(1)
    mean_iou = tf.constant(1)
    update_mean_iou = tf.constant(1)

    if args.segment:
        seg_loss, mean_iou, update_mean_iou = segmentation_loss(seg_logits, seg_gt, config)# 计算目标函数
        the_loss += seg_loss

    if args.detect:
        class_loss, bbox_loss, train_acc = \
            detection_loss(location, confidence, refine_ph, classes_ph, pos_mask)# 计算目标函数
        det_loss = class_loss + bbox_loss
        the_loss += det_loss

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    wd_loss = tf.add_n(regularization_losses)
    tf.summary.scalar('loss\weight_decay', wd_loss)
    the_loss += wd_loss

    tf.summary.scalar('loss\\full', the_loss)
    return the_loss, train_acc, mean_iou, update_mean_iou


def tfrecords_parse_func(serialized, config):
    bboxer = PriorBoxGrid(config)
    features = tf.parse_single_example(
        serialized=serialized,
        features={
            'image/height': tf.FixedLenFeature([], dtype=tf.int64),
            'image/width': tf.FixedLenFeature([], dtype=tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/segmentation/encoded': tf.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/class/label': tf.VarLenFeature(dtype=tf.int64)
        })

    tf_h = features['image/height']
    tf_w = features['image/width']

    im = features['image/encoded']
    im = tf.decode_raw(im, tf.uint8)
    im = tf.reshape(im, (tf_h, tf_w, 3))

    bbox = tf.stack([features['image/object/bbox/%s' % x].values
                     for x in ['ymin', 'xmin', 'ymax', 'xmax']])
    bbox = tf.transpose(bbox)

    gt = features['image/object/class/label'].values

    seg = features['image/segmentation/encoded']
    seg = tf.decode_raw(seg, tf.uint8)
    seg = tf.reshape(seg, (tf_h, tf_w))
    seg = tf.expand_dims(seg, 2)

    im = tf.cast(im, tf.float32)/255
    bbox = yxyx_to_xywh(tf.clip_by_value(bbox, 0.0, 1.0))

    im, bbox, gt, seg = data_augmentation(im, bbox, gt, seg, config)
    inds, cats, refine = bboxer.encode_gt_tf(bbox, gt)

    return im, inds, refine, cats, seg


def read_and_decode(filename_queue, config, random_crop=False, random_clip=False, shuffle_batch=True):
    reader = tf.data.TFRecordDataset(filenames=filename_queue)

    reader = reader.cache()
    reader = reader.shuffle(buffer_size=args.batch_size * args.batch_size, seed=0)
    reader = reader.repeat()

    def tfrecords_parse(serialized):
        return tfrecords_parse_func(serialized, config)

    reader = reader.map(map_func=tfrecords_parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    reader = reader.batch(batch_size=args.batch_size, drop_remainder=True)
    reader = reader.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return reader


def cosine_decay(global_step, steps, learning_rates):
    decayed_lr = tf.constant(learning_rates[0])
    for i, b in enumerate(steps):
        decayed_lr = tf.where(global_step <= b, decayed_lr, learning_rates[i + 1])
    return decayed_lr


def train(net, config):
    #图片预处理
    root_path = args.data_dir
    if args.dataset == 'voc07':
        file_name = 'voc07-trainval'
    if args.dataset == 'voc12-train':
        file_name = 'voc12-train-seg'
    tfrecord_filename = os.path.join(root_path, file_name)
    print("tfrecord_filename = ", tfrecord_filename)
    print("dataset_num_classes = ", dataset_num_classes)
    print("net_config = ", net_config)

    ds_train = read_and_decode(tfrecord_filename, net_config)
    iter_train = ds_train.make_initializable_iterator()
    iter_init_op = iter_train.initializer
    image_ph, inds_ph, refine_ph, classes_ph, seg_gt = iter_train.get_next()

    # with compat.forward_compatibility_horizon(2019, 5, 1):
    net.create_trunk(image_ph)

    if args.detect:
        net.create_multibox_head(dataset_num_classes)
        confidence = net.outputs['confidence']
        location = net.outputs['location']
        tf.summary.histogram('location', location)
        tf.summary.histogram('confidence', confidence)
    else:
        location, confidence = None, None

    if args.segment:
        net.create_segmentation_head(dataset_num_classes)
        seg_logits = net.outputs['segmentation']
        tf.summary.histogram('segmentation', seg_logits)
    else:
        seg_logits = None

    loss, train_acc, mean_iou, update_mean_iou = objective(location, confidence, refine_ph,
                                                           classes_ph, inds_ph, seg_logits,
                                                           seg_gt, config)

    global_step = slim.get_or_create_global_step()
    learning_rate = args.learning_rate

    learning_rates = [args.warmup_lr, learning_rate]
    steps = [args.warmup_step]
    if len(args.lr_decay) > 0:
        for i, step in enumerate(args.lr_decay):
            steps.append(step)
            learning_rates.append(learning_rate*10**(-i-1))

    learning_rate = cosine_decay(tf.to_int32(global_step), steps, learning_rates)

    if args.optimizer == 'adam':
        opt = npu_tf_optimizer(tf.train.AdamOptimizer(learning_rate=learning_rate))
    elif args.optimizer == 'nesterov':
        opt = npu_tf_optimizer(tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True))
    else:
        raise ValueError

    train_vars = tf.trainable_variables()
    print_variables('train', train_vars)

    train_op = slim.learning.create_train_op(
        loss, opt,
        global_step=global_step,
        variables_to_train=train_vars,
        summarize_gradients=True)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000, keep_checkpoint_every_n_hours=1)

    config_npu = tf.ConfigProto()
    custom_op = config_npu.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    # custom_op.parameter_map["mix_compile_mode"].b = True
    config_npu.graph_options.rewrite_options.remapping = RewriterConfig.OFF

    with tf.Session(config=config_npu) as sess:
        init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init_op)

        if args.random_trunk_init:
            print("Training from scratch")
        else:
            init_assign_op, init_feed_dict, init_vars = net.get_imagenet_init(opt)
            print_variables('init from ImageNet', init_vars)
            sess.run(init_assign_op, feed_dict=init_feed_dict)

        #恢复模型
        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if args.ckpt == 0:
                ckpt_to_restore = ckpt.model_checkpoint_path
            else:
                ckpt_to_restore = train_dir+'/model.ckpt-%i' % args.ckpt

            log.info("Restoring model %s..." % ckpt_to_restore)
            saver.restore(sess, ckpt_to_restore)

        starting_step = sess.run(global_step)
        tf.get_default_graph().finalize()

        log.info("Launching prefetch threads")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        log.info("Starting training...")

        sess.run(iter_init_op)

        for step in range(starting_step, args.max_iterations+1):
            start_time = time.time()
            try:
                train_loss, acc, iou, lr = sess.run([train_op, train_acc, mean_iou, learning_rate])
            except (tf.errors.OutOfRangeError, tf.errors.CancelledError):
                break

            duration = time.time() - start_time
            num_examples_per_step = args.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ('step %d, loss = %.2f, acc = %.2f, iou=%f, lr=%.3f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            log.info(format_str % (step, train_loss, acc, iou, -np.log10(lr),
                                   examples_per_sec, sec_per_batch))

            if step % 1000 == 0 and step > 0:
                log.debug("Saving checkpoint...")
                saver.save(sess=sess, save_path=os.path.join(args.result_dir, "model.ckpt"), global_step=step)
                tf.io.write_graph(sess.graph, os.path.join(args.result_dir, "ckpt_npu"), 'graph.pbtxt', as_text=True)
                # modelarts_result2obs(args)

        coord.request_stop()
        coord.join(threads)
        # modelarts_result2obs(args)


def main(argv=None):  # pylint: disable=unused-argument
    if args.trunk == 'resnet50':
        net = ResNet
        depth = 50
    # config = config_resnet_x4, depth = 50, weight_decay = 5e-5
    net = net(config=net_config, depth=depth, training=True, weight_decay=args.weight_decay)

    train(net, net_config)


if __name__ == '__main__':
    tf.app.run()
