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
# Author: Bichen Wu (bichen@berkeley.edu) 03/07/2017

"""Evaluation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

from datetime import datetime
import os.path
import sys
import time

import math
import numpy as np
from six.moves import xrange
import tensorflow as tf
import threading
#import moxing as mox
from config import *
from imdb.kitti import kitti
from utils.util import *
from nets.squeezeSeg import SqueezeSeg
from config.kitti_squeezeSeg_config import kitti_squeezeSeg_config

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_path', '/home/ma-user/modelarts/inputs/data_url_0/',
                           """Root directory of data""")
tf.app.flags.DEFINE_string('ckpt_path', '/home/ma-user/modelarts/outputs/train_url_0/',
                           """Directory where to write event logs and checkpoint. """)
tf.app.flags.DEFINE_string('image_set', 'val', """Can be train, trainval, val, or test""")

# if True, run eval only once to test only one model
# if False,test all the models in the dictionary of checkpoint.
tf.app.flags.DEFINE_boolean('run_once', True, "Whether to run eval only once.""")
tf.app.flags.DEFINE_string('test_model', "model.ckpt-1000", """The model which will be tested.""")

tf.app.flags.DEFINE_string('dataset', 'KITTI', """Currently support KITTI dataset.""")
tf.app.flags.DEFINE_string('eval_dir', '/cache/eval_val/', """Directory where to write event logs """)
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 1, """How often to check if new cpt is saved.""")
tf.app.flags.DEFINE_string('net', 'squeezeSeg', """Neural net architecture.""")


def eval_once(
        saver, ckpt_path, summary_writer, eval_summary_ops, eval_summary_phs, imdb,
        model, data_iterator):
    config = tf.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes(
        "./src/fusion_switch.cfg")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    config.allow_soft_placement = True

    with tf.Session(config=npu_config_proto(config_proto=config)) as sess:

        # Restores from checkpoint
        saver.restore(sess, ckpt_path)
        # Assuming model_checkpoint_path looks something like:
        #   /ckpt_dir/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt_path.split('/')[-1].split('-')[-1]

        mc = model.mc
        mc.DATA_AUGMENTATION = False

        num_images = len(imdb.image_idx)

        _t = {
            'read_and_detect': Timer(),
            'eval': Timer()
        }

        tot_error_rate, tot_rmse, tot_th_correct = 0.0, 0.0, 0.0

        # class-level metrics
        tp_sum = np.zeros(mc.NUM_CLASS)
        fn_sum = np.zeros(mc.NUM_CLASS)
        fp_sum = np.zeros(mc.NUM_CLASS)
        # instance-level metrics
        itp_sum = np.zeros(mc.NUM_CLASS)
        ifn_sum = np.zeros(mc.NUM_CLASS)
        ifp_sum = np.zeros(mc.NUM_CLASS)
        # instance-level object matching metrics
        otp_sum = np.zeros(mc.NUM_CLASS)
        ofn_sum = np.zeros(mc.NUM_CLASS)
        ofp_sum = np.zeros(mc.NUM_CLASS)
        sess.run(data_iterator.initializer)

        for i in xrange(int(num_images / mc.BATCH_SIZE)):
            offset = max((i + 1) * mc.BATCH_SIZE - num_images, 0)

            _t['read_and_detect'].tic()

            op_list = [model.lidar_input, model.lidar_mask, model.label, model.pred_cls]
            lidar_per_batch, lidar_mask_per_batch, label_per_batch, pred_cls = sess.run(op_list)

            _t['read_and_detect'].toc()

            _t['eval'].tic()
            # Evaluation
            iou, tps, fps, fns = evaluate_iou(
                label_per_batch[:mc.BATCH_SIZE - offset],
                pred_cls[:mc.BATCH_SIZE - offset] \
                * np.squeeze(lidar_mask_per_batch[:mc.BATCH_SIZE - offset]),
                mc.NUM_CLASS
            )

            tp_sum += tps
            fn_sum += fns
            fp_sum += fps

            _t['eval'].toc()

            print('detect: {:d}/{:d} read_and_detect: {:.3f}s '
                  'evaluation: {:.3f}s'.format(
                (i + 1) * mc.BATCH_SIZE - offset, num_images,
                _t['read_and_detect'].average_time / mc.BATCH_SIZE,
                _t['eval'].average_time / mc.BATCH_SIZE))

        ious = tp_sum.astype(np.float) / (tp_sum + fn_sum + fp_sum + mc.DENOM_EPSILON)
        pr = tp_sum.astype(np.float) / (tp_sum + fp_sum + mc.DENOM_EPSILON)
        re = tp_sum.astype(np.float) / (tp_sum + fn_sum + mc.DENOM_EPSILON)

        print('Evaluation summary:')
        print('  Timing:')
        print('    read_and_detect: {:.3f}s '.format(
            _t['read_and_detect'].average_time / mc.BATCH_SIZE,
        ))

        eval_sum_feed_dict = {
            eval_summary_phs['Timing/read_and_detect']: _t['read_and_detect'].average_time / mc.BATCH_SIZE,
        }

        print('  Accuracy:')
        for i in range(1, mc.NUM_CLASS):
            #print('    '.format(mc.CLASSES[i]))
            print('{}: Pixel-seg: P: {:.3f}, R: {:.3f}, IoU: {:.3f}'.format(
                mc.CLASSES[i], pr[i], re[i], ious[i]))
            eval_sum_feed_dict[
                eval_summary_phs['Pixel_seg_accuracy/' + mc.CLASSES[i] + '_iou']] = ious[i]
            eval_sum_feed_dict[
                eval_summary_phs['Pixel_seg_accuracy/' + mc.CLASSES[i] + '_precision']] = pr[i]
            eval_sum_feed_dict[
                eval_summary_phs['Pixel_seg_accuracy/' + mc.CLASSES[i] + '_recall']] = re[i]

        eval_summary_str = sess.run(eval_summary_ops, feed_dict=eval_sum_feed_dict)
        for sum_str in eval_summary_str:
            summary_writer.add_summary(sum_str, global_step)
        summary_writer.flush()


def evaluate():
    """Evaluate."""
    assert FLAGS.dataset == 'KITTI', \
        'Currently only supports KITTI dataset'

    with tf.Graph().as_default() as g:

        assert FLAGS.net == 'squeezeSeg', \
            'Selected neural net architecture not supported: {}'.format(FLAGS.net)

        if FLAGS.net == 'squeezeSeg':
            mc = kitti_squeezeSeg_config()
            mc.LOAD_PRETRAINED_MODEL = False
            mc.BATCH_SIZE = 1
            mc.KEEP_PROB = 1.0

        imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)
        data_iterator = imdb.read_batch()
        data_getnext = data_iterator.get_next()

        if FLAGS.net == 'squeezeSeg':
            model = SqueezeSeg(mc, data_getnext)

        eval_summary_ops = []
        eval_summary_phs = {}

        eval_summary_names = [
            'Timing/read_and_detect',
        ]
        for i in range(1, mc.NUM_CLASS):
            eval_summary_names.append('Pixel_seg_accuracy/' + mc.CLASSES[i] + '_iou')
            eval_summary_names.append('Pixel_seg_accuracy/' + mc.CLASSES[i] + '_precision')
            eval_summary_names.append('Pixel_seg_accuracy/' + mc.CLASSES[i] + '_recall')

        for sm in eval_summary_names:
            ph = tf.placeholder(tf.float32)
            eval_summary_phs[sm] = ph
            eval_summary_ops.append(tf.summary.scalar(sm, ph))

        saver = tf.train.Saver(model.model_params)

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        ckpts = set()

        if FLAGS.run_once:
                # When run_once is true, checkpoint_path should point to the exact
                # checkpoint file.
            eval_once(
                saver, os.path.join(FLAGS.ckpt_path, FLAGS.test_model), summary_writer, eval_summary_ops,
                eval_summary_phs, imdb, model, data_iterator)

        else:
                # When run_once is false, checkpoint_path should point to the directory
                # that stores checkpoint files.
            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt_path)
            if ckpt and ckpt.all_model_checkpoint_paths:
                for ckpt_path in ckpt.all_model_checkpoint_paths:
                    eval_once(
                        saver, ckpt_path, summary_writer,
                        eval_summary_ops, eval_summary_phs, imdb, model, data_iterator)
            else:
                print('No checkpoint file found')
                if not FLAGS.run_once:
                    print('Wait {:d}s for new checkpoints to be saved ... '
                              .format(FLAGS.eval_interval_secs))
                    time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    evaluate()


if __name__ == '__main__':
    tf.app.run()
