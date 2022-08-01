# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MixMatch training.
- Ensure class consistency by producing a group of `nu` augmentations of the same image and guessing the label for the
  group.
- Sharpen the target distribution.
- Use the sharpened distribution directly as a smooth label in MixUp.
"""
from npu_bridge.npu_init import *

import functools
import os


# os.system('pip install easydict')
from absl import app
from tensorflow.python.platform import flags
# from absl import flags
from easydict import EasyDict
from libml import layers
from libml import utils
from libml import models
from libml.data_pair import DATASETS
from libml.layers import MixMode
import tensorflow as tf
import sys


FLAGS = flags.FLAGS
flags.DEFINE_float('wd', 0.02, 'Weight decay.')
flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
flags.DEFINE_float('beta', 0.5, 'Mixup beta distribution.')
flags.DEFINE_float('w_match', 100, 'Weight for distribution matching loss.')
flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
FLAGS.set_default('dataset', 'cifar10.3@250-5000')
FLAGS.set_default('batch', 64)
FLAGS.set_default('lr', 0.002)
FLAGS.set_default('train_kimg', 1 << 16)
flags.DEFINE_bool('use_fp16', True, '')
flags.DEFINE_integer('num_gpus', 1, 'gpu number')
#flags.DEFINE_string('obs_dir', 'home/ma-user/modelarts/outputs/train_url_0/', 'obs path')
flags.DEFINE_integer('loss_scale', 999, 'Filter size of convolutions.')


class MixMatch(models.MultiModel):

    def augment(self, x, l, beta, **kwargs):
        assert 0, 'Do not call.'

    def guess_label(self, y, classifier, T, **kwargs):
        del kwargs
        logits_y = [classifier(yi, training=True) for yi in y]
        logits_y = tf.concat(logits_y, 0)
        # Compute predicted probability distribution py.
        p_model_y = tf.reshape(tf.nn.softmax(logits_y), [len(y), -1, self.nclass])
        p_model_y = tf.reduce_mean(p_model_y, axis=0)
        # Compute the target distribution.
        p_target = tf.pow(p_model_y, 1. / T)
        p_target /= tf.reduce_sum(p_target, axis=1, keep_dims=True)
        return EasyDict(p_target=p_target, p_model=p_model_y)

    def model(self, batch, lr, wd, ema, beta, w_match, warmup_kimg=1024, nu=2, mixmode='xxy.yxy', **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')
        y_in = tf.placeholder(tf.float32, [None, nu] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [None], 'labels')
        wd *= lr
        w_match *= tf.clip_by_value(tf.cast(self.step, tf.float32) / (warmup_kimg << 10), 0, 1)
        augment = MixMode(mixmode)
        classifier = functools.partial(self.classifier, **kwargs)

        y = tf.reshape(tf.transpose(y_in, [1, 0, 2, 3, 4]), [-1] + hwc)
        guess = self.guess_label(tf.split(y, nu), classifier, T=0.5, **kwargs)
        ly = tf.stop_gradient(guess.p_target)
        lx = tf.one_hot(l_in, self.nclass)
        xy, labels_xy = augment([x_in] + tf.split(y, nu), [lx] + [ly] * nu, [beta, beta])
        x, y = xy[0], xy[1:]
        labels_x, labels_y = labels_xy[0], tf.concat(labels_xy[1:], 0)
        del xy, labels_xy

        batches = layers.interleave([x] + y, batch)
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        logits = [classifier(batches[0], training=True)]
        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
        for batchi in batches[1:]:
            logits.append(classifier(batchi, training=True))
        logits = layers.interleave(logits, batch)
        logits_x = logits[0]
        logits_y = tf.concat(logits[1:], 0)

        loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_x, logits=logits_x)
        loss_xe = tf.reduce_mean(loss_xe)
        loss_l2u = tf.square(labels_y - tf.nn.softmax(logits_y))
        loss_l2u = tf.reduce_mean(loss_l2u)
        tf.summary.scalar('losses/xe', loss_xe)
        tf.summary.scalar('losses/l2u', loss_l2u)


        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in utils.model_vars('classify') if 'kernel' in v.name])

        def create_optimizer(loss, init_lr, optimizer_type="adam"):
            return tf.train.AdamOptimizer(init_lr)


        #train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe + w_match * loss_l2u, colocate_gradients_with_ops=True)
        #train_op = tf.train.AdamOptimizer(lr)
        # opt = NPULossScaleOptimizer(opt, loss_scale_manager)
        # opt = opt.minimize(self.loss)

        loss_manager = loss_xe + w_match * loss_l2u
        train_op = create_optimizer(loss_manager, lr, "adam")
        if FLAGS.use_fp16 and (FLAGS.loss_scale not in [None, -1]):
            opt_tmp = train_op
            if FLAGS.loss_scale == 0:
                loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2 ** 32, incr_every_n_steps=1000,
                                                                       decr_every_n_nan_or_inf=2, decr_ratio=0.5)
            elif FLAGS.loss_scale >= 1:
                loss_scale_manager = FixedLossScaleManager(loss_scale=FLAGS.loss_scale)
            else:
                raise ValueError("Invalid loss scale: %d" % FLAGS.loss_scale)


            opt = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)

        train_op = opt.minimize(loss_manager, colocate_gradients_with_ops=True)

        
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        # Tuning op: only retrain batch norm.
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        classifier(batches[0], training=True)
        train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])

        return EasyDict(
            x=x_in, y=y_in, label=l_in, train_op=train_op, tune_op=train_bn,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),  # No EMA, for debugging.
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))

def main(argv):
    del argv  # Unused.
    assert FLAGS.nu == 2
    dataset = DATASETS[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = MixMatch(
        os.path.join(FLAGS.train_dir, dataset.name),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        ema=FLAGS.ema,

        beta=FLAGS.beta,
        w_match=FLAGS.w_match,

        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)

    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
#    os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = "1"
#    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["DEVICES_ID"] = "1"
    os.environ["GE_USE_STATIC_MEMORY"]= "1"
    utils.setup_tf()
    app.run(main)

