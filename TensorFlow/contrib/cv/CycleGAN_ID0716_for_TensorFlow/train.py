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
import tensorflow as tf
from model import CycleGAN
from config import make_config
from datetime import datetime
import os
import logging
from utils import ImagePool
import time

FLAGS = tf.flags.FLAGS

# adding some parameters
tf.flags.DEFINE_string('result', '../result', "The result directory where the model checkpoints will be written.")
tf.flags.DEFINE_string('dataset', '../horse2zebra', "dataset path")
tf.flags.DEFINE_string('chip', 'cpu', "Run on which chip, (npu or gpu or cpu)")
tf.flags.DEFINE_string('platform', "linux",
                       "Run on linux/apulis platform. Modelarts Platform has some extra data copy operations")
tf.flags.DEFINE_integer('step_per_epoch', 1000, 'no more imformation here.')
tf.flags.DEFINE_integer('train_epochs', 100, 'no more imformation here.')
tf.flags.DEFINE_string('profiling', '', "deploy The profiling")

# network parameters
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_size', 256, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('lambda1', 10,
                        'weight for forward cycle loss (X->Y->X), default: 10')
tf.flags.DEFINE_integer('lambda2', 10,
                        'weight for backward cycle loss (Y->X->Y), default: 10')
tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
if FLAGS.chip == 'npu':
    from npu_bridge.npu_init import *


def train():
    if FLAGS.load_model is not None:
        checkpoints_dir = os.path.join(FLAGS.result, "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/"))
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = os.path.join(FLAGS.result, "checkpoints/{}".format(current_time))
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = CycleGAN(
            X_train_file=os.path.join(FLAGS.dataset, 'X.tfrecords'),
            Y_train_file=os.path.join(FLAGS.dataset, 'Y.tfrecords'),
            batch_size=FLAGS.batch_size,
            image_size=FLAGS.image_size,
            use_lsgan=FLAGS.use_lsgan,
            norm=FLAGS.norm,
            lambda1=FLAGS.lambda1,
            lambda2=FLAGS.lambda2,
            learning_rate=FLAGS.learning_rate,
            beta1=FLAGS.beta1,
            ngf=FLAGS.ngf
        )
        G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x = cycle_gan.model()
        optimizers = cycle_gan.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()

    config = make_config(FLAGS)
    with tf.Session(graph=graph, config=config) as sess:
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            model_file = tf.train.latest_checkpoint(checkpoints_dir)
            saver.restore(sess, model_file)
            step = int(meta_graph_path.split("-")[2].split(".")[0])
            logging.info("====>>>Restore Model, step is %d.====" % step)
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        fake_Y_pool = ImagePool(FLAGS.pool_size)
        fake_X_pool = ImagePool(FLAGS.pool_size)

        while step <= FLAGS.train_epochs * FLAGS.step_per_epoch:
            fake_y_val, fake_x_val = sess.run([fake_y, fake_x])

            # train
            start_time = time.time()
            _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = (
                sess.run(
                    [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op],
                    feed_dict={cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                               cycle_gan.fake_x: fake_X_pool.query(fake_x_val)}
                )
            )
            train_writer.add_summary(summary, step)
            train_writer.flush()

            if step % FLAGS.step_per_epoch == 0:
                logging.info('-----------Step %d:-------------' % step)
                logging.info('  G_loss   : {}'.format(G_loss_val))
                logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
                logging.info('  F_loss   : {}'.format(F_loss_val))
                logging.info('  D_X_loss : {}'.format(D_X_loss_val))
                logging.info('  Perf     : {}'.format(time.time()-start_time))

            if step % (10 * FLAGS.step_per_epoch) == 0:
                save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)

            step += 1
        save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
        logging.info("Model saved in file: %s" % save_path)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
