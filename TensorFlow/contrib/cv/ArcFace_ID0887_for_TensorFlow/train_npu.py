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


import os
import time
import pickle
import argparse
import numpy as np

import io
import yaml

from scipy import misc

import tensorflow as tf
# import tensorflow.contrib.slim as slim
from tensorflow_core.contrib.mixed_precision.python.loss_scale_manager import ExponentialUpdateLossScaleManager

slim = tf.contrib.slim
from datetime import datetime
import os
from losses.logit_loss import get_logits
from data.classificationDataTool import ClassificationImageData
from model import get_embd
from utils import average_gradients, check_folders, analyze_vars
from evaluate_npu import load_bin, evaluate
# from cfg import make_config
from npu_bridge.npu_init import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, help='path to config file', default='configs/config_ms1m_test1.yaml')
    parser.add_argument("--input_dir", default='/home/test_user02/Arcface/dataset',
                        help="path to folder containing images")
    parser.add_argument("--result", default='/home/test_user02/Arcface/code/result', help="where to put output files")

    parser.add_argument("--num_epochs", type=int, default=1, help="")
    parser.add_argument("--batch_size", type=int, default=16, help="")
    parser.add_argument("--step_per_epoch", type=int, default=1000, help="")
    parser.add_argument("--val_freq", type=int, default=1000, help="")

    parser.add_argument("--obs_dir", default="/",
                        help="obs result path, not need on gpu and apulis platform")
    parser.add_argument("--code_dir", default='/home/test_user02/Arcface/code/ArcfaceCode', help="")
    parser.add_argument("--chip", default="npu", help="Run on which chip, (npu or gpu or cpu)")
    parser.add_argument("--platform", default="apulis",
                        help="Run on linux/apulis/modelarts platform. Modelarts Platform has some extra data copy operations")
    parser.add_argument("--npu_profiling", default=False, type=bool, help="profiling for performance or not")

    return parser.parse_args()


def inference(images, labels, is_training_dropout, is_training_bn, config):
    # print("begin get_embd")
    embds, end_points = get_embd(images, is_training_dropout, is_training_bn, config)
    logits = get_logits(embds, labels, config)
    end_points['logits'] = logits
    return embds, logits, end_points


class Trainer:
    def __init__(self, config, args):
        self.args = args
        self.config = config
        subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
        if args.chip == 'npu':
            self.output_dir = os.path.join(args.result, subdir)
        else:
            self.output_dir = os.path.join(config['output_dir'], subdir)

        self.model_dir = os.path.join(self.output_dir, 'models')
        self.log_dir = os.path.join(self.output_dir, 'log')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.debug_dir = os.path.join(self.output_dir, 'debug')
        check_folders([self.output_dir, self.model_dir, self.log_dir, self.checkpoint_dir, self.debug_dir])
        self.val_log = os.path.join(self.output_dir, 'val_log.txt')

        if args.batch_size is None:
            self.batch_size = config['batch_size']
        else:
            self.batch_size = args.batch_size

        # self.batch_size = args.batch_size
        self.gpu_num = config['gpu_num']
        if self.batch_size % self.gpu_num != 0:
            raise ValueError('batch_size must be a multiple of gpu_num')
        self.image_size = config['image_size']

        if args.num_epochs is None:
            self.epoch_num = config['epoch_num']
        else:
            self.epoch_num = args.num_epochs
        
        if args.step_per_epoch is None:
            self.step_per_epoch = config['step_per_epoch']
        else:
            self.step_per_epoch = args.step_per_epoch

        if args.val_freq is None:
            self.val_freq = config['val_freq']
        else:
            self.val_freq = args.val_freq

        #self.epoch_num = config['epoch_num']
        # self.epoch_num = args.num_epochs
        #self.step_per_epoch = config['step_per_epoch']
        #self.val_freq = config['val_freq']
        self.val_data = config['val_data']
        self.val_bn_train = config['val_bn_train']
        # for k, v in config['val_data'].items():
        #     self.val_data[k] = load_bin(v, self.image_size)
        #     imgs = self.val_data[k][0]
        #     np.save(os.path.join(self.debug_dir, k+'.npy'), imgs[:100])

        with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
            f.write(yaml.dump(self.config))

    def build(self):
        self.train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_dropout')
        self.train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_bn')
        self.global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
        self.inc_op = tf.assign_add(self.global_step, 1, name='increment_global_step')
        scale = int(512.0 / self.batch_size)
        lr_steps = [scale * s for s in self.config['lr_steps']]
        lr_values = [v / scale for v in self.config['lr_values']]
        # lr_steps = self.config['lr_steps']
        self.lr = tf.train.piecewise_constant(self.global_step, boundaries=lr_steps, values=lr_values,
                                              name='lr_schedule')

        cid = ClassificationImageData(img_size=self.image_size, augment_flag=self.config['augment_flag'],
                                      augment_margin=self.config['augment_margin'])
        train_dataset = cid.read_TFRecord(os.path.join(self.args.input_dir,
                                                       self.config['train_data'][0])).shuffle(10000).repeat().batch(
            self.batch_size,
            drop_remainder=True)
        train_iterator = train_dataset.make_one_shot_iterator()
        self.train_images, self.train_labels = train_iterator.get_next()
        self.train_images = tf.identity(self.train_images, 'input_images')
        self.train_labels = tf.identity(self.train_labels, 'labels')

        self.embds, self.logits, self.end_points = inference(self.train_images, self.train_labels,
                                                             self.train_phase_dropout, self.train_phase_bn, self.config)
        self.embds = tf.identity(self.embds, 'embeddings')
        self.inference_loss = slim.losses.sparse_softmax_cross_entropy(logits=self.logits, labels=self.train_labels)
        self.wd_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.train_loss = self.inference_loss + self.wd_loss
        pred = tf.arg_max(tf.nn.softmax(self.logits), dimension=-1, output_type=tf.int64)
        self.train_acc = tf.reduce_mean(tf.cast(tf.equal(pred, self.train_labels), tf.float32))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.MomentumOptimizer(learning_rate=self.lr,
                                                       momentum=self.config['momentum']).minimize(self.train_loss)
        self.train_summary = tf.summary.merge([
            tf.summary.scalar('inference_loss', self.inference_loss),
            tf.summary.scalar('wd_loss', self.wd_loss),
            tf.summary.scalar('train_loss', self.train_loss),
            tf.summary.scalar('train_acc', self.train_acc)
        ])

    def run_embds(self, sess, images):
        batch_num = len(images) // self.batch_size
        left = len(images) % self.batch_size
        embds = []
        for i in range(batch_num):
            image_batch = images[i * self.batch_size: (i + 1) * self.batch_size]
            cur_embd = sess.run(self.embds, feed_dict={self.train_images: image_batch, self.train_phase_dropout: False,
                                                       self.train_phase_bn: False})
            embds += list(cur_embd)
        if left > 0:
            image_batch = np.zeros([self.batch_size, self.image_size, self.image_size, 3])
            image_batch[:left, :, :, :] = images[-left:]
            cur_embd = sess.run(self.embds, feed_dict={self.train_images: image_batch, self.train_phase_dropout: False,
                                                       self.train_phase_bn: False})
            embds += list(cur_embd)[:left]
        return np.array(embds)

    def save_image_label(self, images, labels, step):
        save_dir = os.path.join(self.debug_dir, 'image_by_label')
        for i in range(len(labels)):
            if (labels[i] < 10):
                cur_save_dir = os.path.join(save_dir, str(labels[i]))
                check_folders(cur_save_dir)
                misc.imsave(os.path.join(cur_save_dir, '%d_%d.jpg' % (step, i)), images[i])

    def train(self):
        self.build()
        analyze_vars(tf.trainable_variables(), os.path.join(self.output_dir, 'model_vars.txt'))
        with open(os.path.join(self.output_dir, 'regularizers.txt'), 'w') as f:
            for v in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
                f.write(v.name + '\n')

        from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
        config_proto = tf.ConfigProto()
        custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
        tf_config = npu_config_proto(config_proto=config_proto)

        with tf.Session(config=tf_config) as sess:
            tf.global_variables_initializer().run()
            saver_ckpt = tf.train.Saver()
            saver_best = tf.train.Saver()
            print('begin summary', flush=True)
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            print('begin graph', flush=True)
            tf.io.write_graph(sess.graph_def, self.output_dir, 'graph.pbtxt')

            best_acc = 0
            counter = 0
            print('finish graph', flush=True)
            if config['pretrained_model'] != '':
                saver_ckpt.restore(sess, os.path.join(self.args.code_dir, config['pretrained_model']))
                step = int(os.path.basename(config['pretrained_model']).split('.')[0].split('-')[-1])
                sess.run(tf.assign(self.global_step, step))
                counter = self.global_step.eval(sess)
                print('start step: %d' % counter)
            debug = True
            print('finish pre', flush=True)
            for i in range(self.epoch_num):
                for j in range(self.step_per_epoch):
                    start_time = time.time()
                    # print('begin_run ', flush=True)
                    _, l, l_wd, l_inf, acc, s, _ = sess.run(
                        [self.train_op, self.train_loss, self.wd_loss, self.inference_loss, self.train_acc,
                         self.train_summary, self.inc_op],
                        feed_dict={self.train_phase_dropout: True, self.train_phase_bn: True})
                    counter += 1

                    time_second = time.time() - start_time
                    fps = self.batch_size / time_second
                    if (j % 100 == 0):
                        print(
                            "Epoch:[ %2d/%2d ][ %6d/%6d ] time:%.2f, fps:%0.2f, loss:%.3f (inference:%.3f, wd:%.3f), acc:%.3f"
                            % (i, self.epoch_num, j, self.step_per_epoch, time_second, fps, l, l_inf, l_wd, acc),
                            flush=True)
                    if counter % self.val_freq == 0:
                        saver_ckpt.save(sess, os.path.join(self.checkpoint_dir, 'ckpt-m'), global_step=counter)
                        acc = []
                        with open(self.val_log, 'a') as f:
                            f.write('step: %d\n' % counter)
                            for k, v in self.val_data.items():
                                imgs, imgs_f, issame = load_bin(os.path.join(self.args.input_dir, v),
                                                                self.image_size)
                                embds = self.run_embds(sess, imgs)
                                print("evaluate imgs embeding fished", flush=True)
                                embds_f = self.run_embds(sess, imgs_f)
                                print("evaluate imgs_f embeding fished", flush=True)
                                embds = embds / np.linalg.norm(embds, axis=1,
                                                               keepdims=True) + embds_f / np.linalg.norm(embds_f,
                                                                                                         axis=1,
                                                                                                         keepdims=True)
                                tpr, fpr, acc_mean, acc_std, tar, tar_std, far = evaluate(embds, issame,
                                                                                          far_target=1e-3,
                                                                                          distance_metric=0)
                                print("evaluate fished", flush=True)
                                f.write('eval on %s: acc--%1.5f+-%1.5f, tar--%1.5f+-%1.5f@far=%1.5f\n' % (k, acc_mean,
                                                                                                          acc_std, tar,
                                                                                                          tar_std, far))
                                acc.append(acc_mean)
                            acc = np.mean(np.array(acc))
                            if acc > best_acc:
                                saver_best.save(sess, os.path.join(self.model_dir, 'best-m'), global_step=counter)
                                best_acc = acc
                        if args.platform.lower() == 'modelarts':
                            from help_modelarts import modelarts_result2obs
                            modelarts_result2obs(args)

                saver_ckpt.save(sess, os.path.join(self.checkpoint_dir, 'ckpt-m'), global_step=counter)

            # saver_best.save(sess, os.path.join(self.model_dir, 'best-m'), global_step=counter)
            # saver_ckpt.save(sess, os.path.join(self.checkpoint_dir, 'ckpt-m'), global_step=counter)


if __name__ == '__main__':
    args = parse_args()

    # os.environ["DUMP_GE_GRAPH"] = "2"  # ?    os.environ['SLOG_PRINT_TO_STDOUT'] = "1"
    config = yaml.load(open(os.path.join(args.code_dir, args.config_path), 'r', encoding='utf-8'))
    trainer = Trainer(config, args)
    trainer.train()
    if args.platform.lower() == 'modelarts':
        from help_modelarts import modelarts_result2obs

        modelarts_result2obs(args)
