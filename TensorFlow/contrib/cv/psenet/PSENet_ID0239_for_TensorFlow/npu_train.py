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
# Copyright 2020 Huawei Technologies Co., Ltd
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

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from utils.utils_tool import logger, cfg
from npu_bridge.estimator import npu_ops
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig

tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 8, '')
tf.app.flags.DEFINE_integer('num_readers', 32, '')
tf.app.flags.DEFINE_float('learning_rate', 0.00001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './resnet_train/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_integer('loss_scale', 1024, '')
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')
tf.app.flags.DEFINE_boolean('allow_mix_precision', False, 'whether to allow mix precision')
tf.app.flags.DEFINE_boolean('auto_tune', False, 'whether to autotune')

from nets import model
from utils.data_provider import data_provider
import time

FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))

logger.setLevel(cfg.debug)


def tower_loss(images, seg_maps_gt, training_masks, reuse_variables=None):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        seg_maps_pred = model.model(images, is_training=True)

    model_loss = model.loss(seg_maps_gt, seg_maps_pred, training_masks)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        tf.summary.image('input', images)
        tf.summary.image('seg_map_0_gt', seg_maps_gt[:, :, :, 0:1] * 255)
        tf.summary.image('seg_map_0_pred', seg_maps_pred[:, :, :, 0:1] * 255)
        tf.summary.image('training_masks', training_masks)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads

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

def main(argv=None):
    start1 = time.time()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_seg_maps = tf.placeholder(tf.float32, shape=[None, None, None, 6], name='input_score_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.94,
                                               staircase=True)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    # opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    opt = MixedPrecisionOptimizer(opt, scale=FLAGS.loss_scale)
    from npu_bridge.estimator.npu.npu_optimizer import NPUDistributedOptimizer
    opt = NPUDistributedOptimizer(opt)
    # split
    input_images_split = tf.split(input_images, len(gpus))
    input_seg_maps_split = tf.split(input_seg_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        # with tf.device('/gpu:%d' % gpu_id):
        with tf.name_scope('model_%d' % gpu_id) as scope:
            iis = input_images_split[i]
            isegs = input_seg_maps_split[i]
            itms = input_training_masks_split[i]
            total_loss, model_loss = tower_loss(iis, isegs, itms, reuse_variables)
            batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
            reuse_variables = True

            grads = opt.compute_gradients(total_loss)
            tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    # gpu_options=tf.GPUOptions(allow_growth=True)
    # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训练
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 关闭remap开关
    if FLAGS.allow_mix_precision:
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    if FLAGS.auto_tune:
        custom_op.parameter_map["auto_tune_mode"].s = tf.compat.as_bytes("RL,GA")

    with tf.Session(config=config) as sess:
        if FLAGS.restore:
            logger.info('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            logger.debug(ckpt)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        data_generator = data_provider.get_batch(num_workers=FLAGS.num_readers,
                                                 input_size=FLAGS.input_size,
                                                 batch_size=FLAGS.batch_size_per_gpu * len(gpus))

        start = time.time()
        avg_time_per_step1 = 0
        performs = []
        for step in range(FLAGS.max_steps):
            data = next(data_generator)
            ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict={input_images: data[0],
                                                                                input_seg_maps: data[2],
                                                                                input_training_masks: data[3]})
            if np.isnan(tl):
                logger.error('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                avg_time_per_step1 += (time.time() - start) / FLAGS.max_steps
                avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu * len(gpus)) / (time.time() - start)
                performs.append(float(avg_examples_per_second))
                start = time.time()
                logger.info(
                    'Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                        step, ml, tl, avg_time_per_step, avg_examples_per_second))

            if step % FLAGS.save_checkpoint_steps == 0:
                saver.save(sess, os.path.join(FLAGS.checkpoint_path, 'model.ckpt'), global_step=global_step)

            if step % FLAGS.save_summary_steps == 0:
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data[0],
                                                                                             input_seg_maps: data[2],
                                                                                             input_training_masks: data[
                                                                                                 3]})
                summary_writer.add_summary(summary_str, global_step=step)
    print("Final Train Accuracy", tl)
    E2Etime = time.time() - start1
    print("E2E Training Duration sec", E2Etime)
    print("avg time per step", avg_time_per_step1)
    print("FPS {:.2f}".format(sum(performs) / len(performs)))


if __name__ == '__main__':
    tf.app.run()