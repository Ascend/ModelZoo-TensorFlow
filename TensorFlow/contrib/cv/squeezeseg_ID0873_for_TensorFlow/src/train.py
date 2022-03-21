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
# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Train"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from npu_bridge.npu_init import *

import argparse
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
tf.app.flags.DEFINE_string('ckpt_path', './outputs',
                           """Directory where to write event logs and checkpoint. """)
tf.app.flags.DEFINE_integer('max_steps', 2000, """Maximum number of batches to run.""")
tf.app.flags.DEFINE_integer('summary_step', 100
                            , """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 1000, """Number of steps to save summary.""")

tf.app.flags.DEFINE_string('dataset', 'KITTI', """Currently only support KITTI dataset.""")
tf.app.flags.DEFINE_string('image_set', 'train', """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('net', 'squeezeSeg', """Neural net architecture. """)

pretrained_model_path=os.path.join(FLAGS.data_path,"SqueezeNet/squeezenet_v1.1.pkl")
tf.app.flags.DEFINE_string('pretrained_model_path', pretrained_model_path,
                           """Path to the pretrained model.""")


def train():
    """Train SqueezeSeg model"""
    assert FLAGS.dataset == 'KITTI', \
        'Currently only support KITTI dataset'

    with tf.Graph().as_default():

        assert FLAGS.net == 'squeezeSeg', \
            'Selected neural net architecture not supported: {}'.format(FLAGS.net)
        if FLAGS.net == 'squeezeSeg':
            mc = kitti_squeezeSeg_config()
            mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
        imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)
        data_iterator = imdb.read_batch()
        data_getnext = data_iterator.get_next()

        if FLAGS.net == 'squeezeSeg':
            model = SqueezeSeg(mc, data_getnext)

        # save model size, flops, activations by layers
        sava_path = os.path.join(FLAGS.ckpt_path, 'model_metrics.txt')
        if not os.path.exists(sava_path):
            open(sava_path, "a")
        with open(sava_path, 'w') as f:
            f.write('Number of parameter by layer:\n')
            count = 0
            for c in model.model_size_counter:
                f.write('\t{}: {}\n'.format(c[0], c[1]))
                count += c[1]
            f.write('\ttotal: {}\n'.format(count))

            count = 0
            f.write('\nActivation size by layer:\n')
            for c in model.activation_counter:
                f.write('\t{}: {}\n'.format(c[0], c[1]))
                count += c[1]
            f.write('\ttotal: {}\n'.format(count))

            count = 0
            f.write('\nNumber of flops by layer:\n')
            for c in model.flop_counter:
                f.write('\t{}: {}\n'.format(c[0], c[1]))
                count += c[1]
            f.write('\ttotal: {}\n'.format(count))
        f.close()
        print('Model statistics saved to {}.'.format(
            os.path.join(FLAGS.ckpt_path, 'model_metrics.txt')))

        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.summary.merge_all()
        init = tf.initialize_all_variables()

        config = tf.ConfigProto()
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["enable_data_pre_proc"].b = True
        custom_op.parameter_map["fusion_switch_file"].s = tf.compat.as_bytes("./src/fusion_switch.cfg")
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
        config.allow_soft_placement = True
        sess = tf.Session(config=npu_config_proto(config_proto=config))
        sess.run(init)
        summary_writer = tf.summary.FileWriter(FLAGS.ckpt_path, sess.graph)
        sess.run(data_iterator.initializer)

        try:
            for step in xrange(FLAGS.max_steps):
                start_time = time.time()

                if step % FLAGS.summary_step == 0 or step == FLAGS.max_steps - 1:
                    op_list = [
                        model.lidar_input, model.lidar_mask, model.label, model.train_op,
                        model.loss, model.pred_cls, summary_op
                    ]

                    lidar_per_batch, lidar_mask_per_batch, label_per_batch, \
                    _, loss_value, pred_cls, summary_str = sess.run(op_list)

                    label_image = visualize_seg(label_per_batch[:6, :, :], mc)
                    pred_image = visualize_seg(pred_cls[:6, :, :], mc)

                    # Run evaluation on the batch
                    ious, _, _, _ = evaluate_iou(
                        label_per_batch, pred_cls * np.squeeze(lidar_mask_per_batch),
                        mc.NUM_CLASS)

                    feed_dict = {}
                    # Assume that class-0 is the background class
                    for i in range(1, mc.NUM_CLASS):
                        feed_dict[model.iou_summary_placeholders[i]] = ious[i]

                    iou_summary_list = sess.run(model.iou_summary_ops[1:], feed_dict)

                    # Run visualization
                    viz_op_list = [model.show_label, model.show_depth_img, model.show_pred]
                    viz_summary_list = sess.run(
                        viz_op_list,
                        feed_dict={
                            model.depth_image_to_show: lidar_per_batch[:6, :, :, [4]],
                            model.label_to_show: label_image,
                            model.pred_image_to_show: pred_image,
                        }
                    )

                    # Add summaries
                    summary_writer.add_summary(summary_str, step)

                    for sum_str in iou_summary_list:
                        summary_writer.add_summary(sum_str, step)

                    for viz_sum in viz_summary_list:
                        summary_writer.add_summary(viz_sum, step)

                    # force tensorflow to synchronise summaries
                    summary_writer.flush()

                else:
                    _, loss_value = sess.run([model.train_op, model.loss])
                duration = time.time() - start_time

                assert not np.isnan(loss_value), \
                    'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
                    'class_loss: {}'.format(loss_value, conf_loss, bbox_loss,
                                            class_loss)

                if step % 1 == 0:
                    num_images_per_step = mc.BATCH_SIZE
                    images_per_sec = num_images_per_step / duration
                    sec_per_batch = float(duration)
                    format_str = ('%s: step %d, loss = %.2f %.1f images/sec; %.3f '
                                  'sec/batch')
                    print(format_str % (datetime.now(), step, loss_value,
                                        images_per_sec, sec_per_batch))
                    sys.stdout.flush()

                # Save the model checkpoint periodically.
                if step % FLAGS.checkpoint_step == 0 or step == FLAGS.max_steps - 1:
                    checkpoint_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except KeyboardInterrupt:
            print('Interrupted')


def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
