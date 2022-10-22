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

import os
import tensorflow as tf
import functools
import numpy as np
from tensorflow.python.tools import freeze_graph
from libml import models
from libml import layers
from absl import flags, app
from libml import data, utils
from easydict import EasyDict
from libml.data_pair import DATASETS
from libml.layers import MixMode

FLAGS = flags.FLAGS


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

        train_op = tf.train.AdamOptimizer(lr).minimize(loss_xe + w_match * loss_l2u, colocate_gradients_with_ops=True)
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
    dataset = DATASETS[FLAGS.dataset]()
    print('dataset:', dataset)
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

    input_folder = os.path.exists("./mix_model/input_bin_{0:02d}".format(FLAGS.batchsize))
    output_folder = os.path.exists("./mix_model/output_label_{0:02d}".format(FLAGS.batchsize))
    if not input_folder:
        os.makedirs("./mix_model/input_bin_{0:02d}".format(FLAGS.batchsize))
    if not output_folder:
        os.makedirs("./mix_model/output_label_{0:02d}".format(FLAGS.batchsize))
    with tf.Session(config=utils.get_config()) as sess:
        model.session = sess
        model.cache_eval()
        images, labels = model.tmp.cache['test']
        for j in range(0,len(labels),FLAGS.batchsize):
            x = images[j:j+FLAGS.batchsize]
            label = labels[j:j+FLAGS.batchsize]
            np.array(x).tofile("./mix_model/input_bin_{0:02d}/{1:05d}.bin".format(FLAGS.batchsize,j))
            np.savetxt("./mix_model/output_label_{0:02d}/{1:05d}.txt".format(FLAGS.batchsize,j),label)


if __name__ == '__main__':
    flags.DEFINE_float('wd', 0.02, 'Weight decay.')
    flags.DEFINE_float('ema', 0.999, 'Exponential moving average of params.')
    flags.DEFINE_float('beta', 0.5, 'Mixup beta distribution.')
    flags.DEFINE_float('w_match', 100, 'Weight for distribution matching loss.')
    flags.DEFINE_integer('scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer('batchsize',1,'the batch number of IMG in one bin file')
    app.run(main)
    # python convert_img_2_bin.py --batchsize=1