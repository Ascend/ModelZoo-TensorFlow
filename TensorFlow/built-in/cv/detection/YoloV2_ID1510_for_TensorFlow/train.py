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
from npu_bridge.npu_init import *
import tensorflow as tf
import datetime
import os
import argparse
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import *
import time

class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg(cfg)

        self.variable_to_restore = tf.global_variables()
        self.restorer = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = npu_tf_optimizer(tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)).minimize(
            self.net.total_loss, global_step=self.global_step)
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        self.averages_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.averages_op)

        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = "NpuOptimizer"
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["graph_memory_max_size"].s = tf.compat.as_bytes(str(16*1024*1024*1024))
        custom_op.parameter_map["variable_memory_max_size"].s = tf.compat.as_bytes(str(15*1024*1024*1024))
        self.sess = tf.Session(config=npu_config_proto(config_proto=config))
        self.sess.run(tf.global_variables_initializer())

        if self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_file)
            self.restorer.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)

    def train(self):

        train_timer = Timer()
        load_timer = Timer()
        trainset = tf.data.Dataset.from_generator(
            self.data, output_types=(tf.string, tf.float32, tf.bool)
        )
        trainset = trainset.map(lambda x, y, z:tf.py_func(image_read, inp=[x, y, z], Tout=[tf.string, tf.float32, tf.bool]), num_parallel_calls=16)
        trainset = trainset.map(proc_func, num_parallel_calls=16)
        batch_data = trainset.batch(cfg.BATCH_SIZE)
        batch_data = batch_data.prefetch(2)
        iterator = batch_data.make_initializable_iterator()
        next_element = iterator.get_next()
        init_op = iterator.initializer
        self.sess.run(init_op)

        for step in range(1, self.max_iter + 1):

            #load_timer.tic()
            load_start = time.time()
            images, labels = self.sess.run(next_element)
            #load_timer.toc()
            load_duration = time.time() - load_start
            feed_dict = {self.net.images: images, self.net.labels: labels}

            train_start = time.time()
            summary_str, loss, _ = self.sess.run(
                [self.summary_op, self.net.total_loss, self.train_op],
                feed_dict=feed_dict)
            train_duration = time.time() - train_start

            log_str = ('{} Epoch: {}, Step: {}, Learning rate: {},'
                ' Loss: {:5.3f}\nSpeed: {:.3f}s/iter,'
                ' Load: {:.3f}s/iter').format(
                datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                self.data.epoch,
                int(step),
                round(self.learning_rate.eval(session=self.sess), 6),
                loss,
                train_duration,
                load_duration,
                )
            print(log_str)

            self.writer.add_summary(summary_str, step)

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(self.sess, self.ckpt_file,
                                global_step=self.global_step)

    def save_cfg(self, cfg):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def update_config_paths(args):
    cfg.DATA_PATH = args.data_dir
    cfg.BATCH_SIZE = args.batch_size
    cfg.MAX_ITER = args.max_iter
    cfg.LEARNING_RATE = args.learning_rate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--max_iter', default=15000, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    update_config_paths(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet()
    pascal = pascal_voc('train')

    solver = Solver(yolo, pascal)

    print('Start training ...')
    solver.train()
    print('Done training.')

if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()

