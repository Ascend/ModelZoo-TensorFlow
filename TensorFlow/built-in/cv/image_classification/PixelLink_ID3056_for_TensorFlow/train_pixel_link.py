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
# test code to make sure the ground truth calculation and data batch works well.
from npu_bridge.npu_init import *
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf  # test
from tensorflow.python.ops import control_flow_ops
from npu_bridge.estimator.npu import npu_loss_scale_manager as lsm_lib
from datasets import dataset_factory

from nets import pixel_link_symbol
import util
import pixel_link
import glob
from preprocessing import ssd_vgg_preprocessing
import collections
from tensorflow.python.training import checkpoint_management
import time

slim = tf.contrib.slim
import config

tf.enable_control_flow_v2()
tf.enable_resource_variables()

# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_string('train_dir', './train_dir',
                           'the path to store checkpoints and eventfiles for summaries')

tf.app.flags.DEFINE_string('checkpoint_path', None,
                           'the path of pretrained model to be used. If there are checkpoints in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', -1,
                          'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')

tf.app.flags.DEFINE_integer('batch_size', 24, 'The number of samples in each batch.')   #24
tf.app.flags.DEFINE_integer('num_gpus', 1, 'The number of gpus can be used.')
tf.app.flags.DEFINE_integer('max_number_of_steps', 60000, 'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('log_every_n_steps', 1, 'log frequency')
tf.app.flags.DEFINE_integer('log_interval_steps', 10000, 'ckpt save frequency')
tf.app.flags.DEFINE_bool("ignore_missing_vars", False, '')
tf.app.flags.DEFINE_bool("dynamic_loss_scale", False, '')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None, 'checkpoint_exclude_scopes')

# =========================================================================== #
# Optimizer configs.
# =========================================================================== #
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_bool('using_moving_average', True, 'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay rate of ExponentionalMovingAverage')

# =========================================================================== #
# I/O and preprocessing Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_readers', 1,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 24,
    'The number of threads used to create the batches.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'icdar2015', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
# tf.app.flags.DEFINE_string(
#     'dataset_dir', './datasets/pixel_link/ICDAR2015', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'dataset_dir', './datasets/pixel_link/ICDAR2015', 'The directory where the dataset files are stored.')
# tf.app.flags.DEFINE_string(
#     'dataset_dir', './datasets/pixel_link/icdar2015/ICDAR2015', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('train_image_width', 512, 'Train image size')
tf.app.flags.DEFINE_integer('train_image_height', 512, 'Train image size')

FLAGS = tf.app.flags.FLAGS

InputEndpoints = collections.namedtuple(
    'InputEndpoints', ['b_image', 'b_pixel_cls_label', 'b_pixel_cls_weight', 'b_pixel_link_label', 'b_pixel_link_weight'])

def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.train_image_height, FLAGS.train_image_width)

    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    #util.init_logger(
    #    log_file='log_train_pixel_link_%d_%d.log' % image_shape,
    #    log_path=FLAGS.train_dir, stdout=False, mode='a')

    config.load_config(FLAGS.train_dir)

    config.init_config(image_shape,
                       batch_size=FLAGS.batch_size,
                       weight_decay=FLAGS.weight_decay,
                       num_gpus=FLAGS.num_gpus
                       )

    batch_size = config.batch_size
    batch_size_per_gpu = config.batch_size_per_gpu

    tf.summary.scalar('batch_size', batch_size)
    tf.summary.scalar('batch_size_per_gpu', batch_size_per_gpu)

    util.proc.set_proc_name('train_pixel_link_on' + '_' + FLAGS.dataset_name)

    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    config.print_config(FLAGS, dataset)
    return dataset

def parse_record_v1(example_string, dataset):
    # tfrecord中存储的map
    feature_dict = tf.parse_single_example(example_string, dataset.keys_to_features)
    # 数据预处理方式与input_producer方式保持一致
    outputs = []
    for item in dataset.items_to_handlers:
        handler = dataset.items_to_handlers[item]
        keys_to_tensors = {key: feature_dict[key] for key in handler.keys}
        outputs.append(handler.tensors_to_item(keys_to_tensors))
    image = outputs[0]
    shape = outputs[1]
    filename = outputs[2]
    gbboxes = outputs[3]
    x1, x2, x3, x4 = outputs[4], outputs[5], outputs[6], outputs[7]
    y1, y2, y3, y4 = outputs[8], outputs[9], outputs[10], outputs[11]
    glabel = outputs[12]


    gxs = tf.transpose(tf.stack([x1, x2, x3, x4]))  # shape = (N, 4)
    gys = tf.transpose(tf.stack([y1, y2, y3, y4]))
    image = tf.identity(image, 'input_image')

    # Pre-processing image, labels and bboxes.
    image, glabel, gbboxes, gxs, gys = \
        ssd_vgg_preprocessing.preprocess_image(
            image, glabel, gbboxes, gxs, gys,
            out_shape=config.train_image_shape,
            data_format=config.data_format,
            use_rotation=config.use_rotation,
            is_training=True)
    image = tf.identity(image, 'processed_image')

    # calculate ground truth
    pixel_cls_label, pixel_cls_weight, \
    pixel_link_label, pixel_link_weight = \
        pixel_link.tf_cal_gt_for_single_image(gxs, gys, glabel)

    return image, pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight


def create_dataset_batch_queue(dataset):
    from preprocessing import ssd_vgg_preprocessing

    with tf.device('/cpu:0'):

        with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
            tf_list = glob.glob(dataset.data_sources)
            raw_dataset = tf.data.TFRecordDataset(tf_list).repeat()
            datasets = raw_dataset.map(lambda value: parse_record_v1(value, dataset),
                                       num_parallel_calls=FLAGS.num_preprocessing_threads).shuffle(300 * config.batch_size)
            datasets = datasets.batch(int(config.batch_size_per_gpu), drop_remainder=True)
            iterator = tf.data.make_initializable_iterator(datasets)
            b_image, b_pixel_cls_label, b_pixel_cls_weight, \
            b_pixel_link_label, b_pixel_link_weight = iterator.get_next()

    return InputEndpoints(
        b_image=b_image,
        b_pixel_cls_label=b_pixel_cls_label,
        b_pixel_cls_weight=b_pixel_cls_weight,
        b_pixel_link_label=b_pixel_link_label,
        b_pixel_link_weight=b_pixel_link_weight), iterator


def sum_gradients(clone_grads, scaled_loss):
    averaged_grads = []
    for grad_and_vars in zip(*clone_grads):
        grads = []
        var = grad_and_vars[0][1]
        try:
            for g, v in grad_and_vars:
                assert v == var
                if FLAGS.dynamic_loss_scale:
                    grads.append(g/scaled_loss)
                else:
                    grads.append(g)
            grad = tf.add_n(grads, name=v.op.name + '_summed_gradients')
        except:
            import pdb
            pdb.set_trace()

        averaged_grads.append((grad, v))

    #         tf.summary.histogram("variables_and_gradients_" + grad.op.name, grad)
    #         tf.summary.histogram("variables_and_gradients_" + v.op.name, v)
    #         tf.summary.scalar("variables_and_gradients_" + grad.op.name+\
    #               '_mean/var_mean', tf.reduce_mean(grad)/tf.reduce_mean(var))
    #         tf.summary.scalar("variables_and_gradients_" + v.op.name+'_mean',tf.reduce_mean(var))
    return averaged_grads


def create_clones(batch_queue):
    with tf.device('/cpu:0'):

        global_step = tf.train.get_or_create_global_step()
        # learning_rate = tf.constant(FLAGS.learning_rate, name='learning_rate')

        lr_1 = FLAGS.learning_rate
        lr_2 = lr_1 * 10
        lrs = [lr_1, lr_2]
        boundaries = [100]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, lrs)

        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               momentum=FLAGS.momentum, name='Momentum')

        if FLAGS.dynamic_loss_scale:
            loss_scale_manager = lsm_lib.ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=1, decr_ratio=0.5)
            optimizer = NPUOptimizer(optimizer, loss_scale_manager, is_distributed=False,
                                 is_loss_scale=True, is_tailing_optimization=False)
            scaled_loss = loss_scale_manager.get_loss_scale()
        else:
            scaled_loss = None
        tf.summary.scalar('learning_rate', learning_rate)
    # place clones
    pixel_link_loss = 0;  # for summary only
    gradients = []
    for clone_idx, gpu in enumerate(config.gpus):
        do_summary = clone_idx == 0  # only summary on the first clone
        reuse = clone_idx > 0
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            with tf.name_scope(config.clone_scopes[clone_idx]) as clone_scope:
                with tf.device('/cpu:0') as clone_device:
                    # b_image, b_pixel_cls_label, b_pixel_cls_weight, \
                    # b_pixel_link_label, b_pixel_link_weight = batch_queue.dequeue()
                    b_image, b_pixel_cls_label, b_pixel_cls_weight, \
                    b_pixel_link_label, b_pixel_link_weight = batch_queue.b_image, batch_queue.b_pixel_cls_label, \
                                                              batch_queue.b_pixel_cls_weight, batch_queue.b_pixel_link_label, \
                                                              batch_queue.b_pixel_link_weight
                    # build model and loss
                    net = pixel_link_symbol.PixelLinkNet(b_image, is_training=True)
                    net.build_loss(
                        pixel_cls_labels=b_pixel_cls_label,
                        pixel_cls_weights=b_pixel_cls_weight,
                        pixel_link_labels=b_pixel_link_label,
                        pixel_link_weights=b_pixel_link_weight,
                        do_summary=do_summary)

                    # gather losses
                    losses = tf.get_collection(tf.GraphKeys.LOSSES, clone_scope)
                    assert len(losses) == 2
                    total_clone_loss = tf.add_n(losses) / config.num_clones
                    pixel_link_loss += total_clone_loss

                    # gather regularization loss and add to clone_0 only
                    if clone_idx == 0:
                        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        total_clone_loss = total_clone_loss + regularization_loss

                    # compute clone gradients
                    if FLAGS.dynamic_loss_scale:
                        clone_gradients = optimizer.compute_gradients(total_clone_loss * scaled_loss)
                    else:
                        clone_gradients = optimizer.compute_gradients(total_clone_loss)
                    gradients.append(clone_gradients)

    tf.summary.scalar('pixel_link_loss', pixel_link_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)

    # add all gradients together
    # note that the gradients do not need to be averaged, because the average operation has been done on loss.
    averaged_gradients = sum_gradients(gradients, scaled_loss)

    apply_grad_op = optimizer.apply_gradients(averaged_gradients, global_step=global_step)

    train_ops = [apply_grad_op]

    bn_update_op = util.tf.get_update_op()
    if bn_update_op is not None:
        train_ops.append(bn_update_op)

    # moving average
    if FLAGS.using_moving_average:
        tf.logging.info('using moving average in training, \
        with decay = %f' % (FLAGS.moving_average_decay))
        ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([apply_grad_op]):  # ema after updating
            train_ops.append(tf.group(ema_op))

    train_op = control_flow_ops.with_dependencies(train_ops, pixel_link_loss, name='train_op')
    return train_op


def train(train_op, iterator):
    summary_op = tf.summary.merge_all()
    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

    init_fn = util.tf.get_init_fn(checkpoint_path=FLAGS.checkpoint_path, train_dir=FLAGS.train_dir,
                                  ignore_missing_vars=FLAGS.ignore_missing_vars,
                                  checkpoint_exclude_scopes=FLAGS.checkpoint_exclude_scopes)

    saver = tf.train.Saver(max_to_keep=10, write_version=2)
    slim.learning.train(
        train_op,
        logdir=FLAGS.train_dir,
        init_fn=init_fn,
        summary_op=summary_op,
        number_of_steps=FLAGS.max_number_of_steps,
        log_every_n_steps=FLAGS.log_every_n_steps,
        save_summaries_secs=30,
        saver=saver,
        # save_interval_secs = 1200,
        save_interval_secs=10800,
        session_config=sess_config
    )

def my_train(train_op, iterator):
    summary_op = tf.summary.merge_all()
    # sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)  #edit 2022/7/14
    sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
    custom_op =  sess_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    saver = tf.train.Saver(max_to_keep=10, write_version=2)
    with tf.Session(config=npu_config_proto(config_proto=sess_config)) as session:
        tf.global_variables_initializer().run()
        iterator.initializer.run()
        number_of_steps = FLAGS.max_number_of_steps
        log_interval_steps = FLAGS.log_interval_steps
        log_every_n_steps = FLAGS.log_every_n_steps
        logdir = FLAGS.train_dir
        step_index = 0
        global_step = tf.train.get_or_create_global_step()
        checkpoint_dir = FLAGS.checkpoint_path
        if checkpoint_dir:
            ckpt = checkpoint_management.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                step_index = int(ckpt.model_checkpoint_path.split("-")[-1])
                print("ckpt.model_checkpoint_path:{}".format(ckpt.model_checkpoint_path))
                saver.restore(session, ckpt.model_checkpoint_path)
        while True:
            if number_of_steps is not None and step_index >= number_of_steps:
                break
            # step_index += 1
            step_start = time.time()

            total_loss, step_index = session.run((train_op, global_step))
            step_end = time.time()
            print("global step {:d}: loss = {:.4f} ({:.3f} sec/step)".format(step_index, total_loss, step_end - step_start))
            if step_index % log_interval_steps == 0:
                saver.save(session, logdir + os.sep + 'model.ckpt', global_step=step_index)




def main(_):
    # The choice of return dataset object via initialization method maybe confusing, 
    # but I need to print all configurations in this method, including dataset information. 
    dataset = config_initialization()

    batch_queue, iterator = create_dataset_batch_queue(dataset)
    train_op = create_clones(batch_queue)
    my_train(train_op, iterator)


if __name__ == '__main__':
    tf.app.run()